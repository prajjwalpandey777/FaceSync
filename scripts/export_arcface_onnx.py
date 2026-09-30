"""Convert YOUR trained ArcFace model (weights/backbone.pth) to ONNX so it can run inside the browser.

    python scripts/export_arcface_onnx.py
    python scripts/export_arcface_onnx.py --checkpoint weights/backbone.pth --out frontend/models/arcface_r100.onnx

What this does and does NOT do
  * It loads your checkpoint with the project's own strict loader (backbone/load_model.py), exactly as the
    desktop app and the old server did: same architecture, same weights, nothing retrained or altered.
  * It writes the same network in the ONNX file format (weights stored as float32, so the numbers are identical).
  * It then runs the PyTorch model and the ONNX file on the same random inputs and refuses to finish
    unless they agree (cosine similarity >= 0.9999), so you know the conversion is faithful.

Options
  --compact   store the weights as float16 inside the file (about half the download, ~130 MB instead of ~261 MB).
              The browser converts them back to float32 at load time. Differences are tiny but not zero, so it is
              OFF by default. If you use it, the script checks the result the same way.
Needs:  pip install torch onnx onnxruntime numpy
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "weights" / "backbone.pth"))
    ap.add_argument("--out", default=str(ROOT / "frontend" / "models" / "arcface_r100.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--compact", action="store_true", help="store weights as float16 (smaller download)")
    args = ap.parse_args()

    import torch
    import onnx
    import onnxruntime as ort
    from backbone.load_model import load_iresnet100

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}\nPut your trained weights at weights/backbone.pth (or use --checkpoint).")
        return 2

    print(f"[1/4] Loading your model from {ckpt} (strict check: architecture and weights must match exactly)")
    model = load_iresnet100(checkpoint_path=ckpt, device="cpu").eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 112, 112)

    print(f"[2/4] Exporting to ONNX (opset {args.opset}) -> {out}")
    kwargs = dict(
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)   # classic exporter: most compatible
        except TypeError:                                                        # older torch has no `dynamo` flag
            torch.onnx.export(model, dummy, str(out), **kwargs)

    # Some exporters put the weights in a separate ".data" file; fold everything into ONE file for the browser.
    proto = onnx.load(str(out), load_external_data=True)
    if args.compact:
        proto = to_fp16_storage(proto)
    for ext in (out.with_suffix(out.suffix + ".data"),):
        if ext.exists():
            ext.unlink()
    onnx.save_model(proto, str(out), save_as_external_data=False)
    onnx.checker.check_model(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"      wrote {out.name}: {size_mb:.0f} MB")

    print("[3/4] Verifying: PyTorch model vs ONNX file on random inputs")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst_cos, worst_rel = 1.0, 0.0
    g = torch.Generator().manual_seed(1234)
    for n in (1, 3):
        x = torch.randn(n, 3, 112, 112, generator=g)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        for a, b in zip(ref, got):
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            worst_cos = min(worst_cos, cos)
        worst_rel = max(worst_rel, float(np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-12)))
    print(f"      worst cosine similarity = {worst_cos:.6f}   worst relative difference = {worst_rel:.2e}")
    if worst_cos < 0.9999:
        print("ERROR: the ONNX file does not match your PyTorch model closely enough. Do not use it.")
        return 3

    print("[4/4] Done.")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"      sha256 {sha}\n\nNext: follow DEPLOYMENT.md step 6 to host this file and give its URL to the website.")
    return 0


def to_fp16_storage(proto):
    """Keep fp16 copies of the big weight tensors and cast them back to fp32 inside the graph.
    Compute still happens in float32 (works on every browser backend); only the stored file is smaller."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = proto.graph
    new_inits, cast_nodes = [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims or [1])) >= 1024:
            arr = numpy_helper.to_array(init)
            small = numpy_helper.from_array(arr.astype(np.float16), name=init.name + "__fp16")
            new_inits.append(small)
            cast_nodes.append(helper.make_node("Cast", [small.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            new_inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + nodes)       # casts first keeps the graph topologically sorted
    return proto


if __name__ == "__main__":
    sys.exit(main())
