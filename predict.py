"""Prediction script for ArcFace face recognition system."""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import torch.nn.functional as F

from backbone.load_model import load_iresnet100
from face_detector import FaceDetector
from anti_spoof import MiniFASNetV2AntiSpoof


DEFAULT_DB_PATH = PROJECT_ROOT / "weights" / "face_db.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "backbone.pth"
DEFAULT_THRESHOLD = 0.45
DEFAULT_ANTI_SPOOF_MODEL = PROJECT_ROOT / "weights" / "2.7_80x80_MiniFASNetV2.onnx"
DEFAULT_ANTI_SPOOF_THRESHOLD = 0.60


def load_database(db_path: Path) -> Dict:
    """Load enrolled face identity database."""
    if not db_path.is_file():
        print(f"Error: Face database not found at {db_path}.")
        print("Run 'python build_database.py' first to create the database.")
        sys.exit(1)

    db = torch.load(db_path, map_location="cpu")
    return db


def match_face_embedding(
    face_emb: torch.Tensor,
    db: Dict,
    threshold: float,
) -> Tuple[str, float, str]:
    """Compare face embedding against enrolled identities in the database.

    Returns:
        (best_person_name, similarity_score, status)
    """
    best_name = "Unknown"
    best_similarity = -1.0

    # Ensure query embedding is 2D unit vector on CPU: [1, 512]
    face_emb = face_emb.cpu()
    if face_emb.ndim == 1:
        face_emb = face_emb.unsqueeze(0)
    face_emb = F.normalize(face_emb, p=2, dim=1)

    for person_name, record in db["persons"].items():
        gallery_embs = record["embeddings"]  # [N, 512]
        # Pairwise cosine similarities to all enrolled photos of this person
        sims = torch.mm(face_emb, gallery_embs.t()).squeeze(0)  # [N]
        # Maximum similarity among enrolled photos
        max_sim = torch.max(sims).item()

        if max_sim > best_similarity:
            best_similarity = max_sim
            best_name = person_name

    status = "MATCH" if best_similarity >= threshold else "UNKNOWN"
    if status == "UNKNOWN":
        best_name = "Unknown"

    return best_name, max(0.0, best_similarity), status


def predict_image(
    image_path: Path,
    db_path: Path = DEFAULT_DB_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    threshold: Optional[float] = None,
    all_faces: bool = False,
    device: Optional[str] = None,
    detector_backend: str = "auto",
    anti_spoof: bool = True,
    anti_spoof_threshold: float = DEFAULT_ANTI_SPOOF_THRESHOLD,
) -> None:
    """Recognize faces in a test image and print results in the requested format."""
    image_path = Path(image_path)
    if not image_path.is_file():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Load database
    db = load_database(db_path)
    eff_threshold = threshold if threshold is not None else db.get("default_threshold", DEFAULT_THRESHOLD)

    # Load model and detector
    model = load_iresnet100(checkpoint_path=checkpoint_path, device=device)
    target_device = next(model.parameters()).device
    detector = FaceDetector(backend=detector_backend)

    # Load anti-spoof model if enabled
    spoof_detector: Optional[MiniFASNetV2AntiSpoof] = None
    if anti_spoof:
        spoof_detector = MiniFASNetV2AntiSpoof(
            model_path=DEFAULT_ANTI_SPOOF_MODEL,
            threshold=anti_spoof_threshold,
        )

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    faces = detector.detect_faces(img, device=target_device)
    if not faces:
        print("Person: None")
        print("Similarity: 0.00")
        print("Status: NO_FACE_DETECTED")
        return

    if all_faces and len(faces) > 1:
        print(f"Detected {len(faces)} faces in image:")
        for idx, face in enumerate(faces, start=1):
            print(f"\nFace #{idx} (bbox: {face['bbox']}):")
            if spoof_detector is not None:
                is_real, liveness_score, _ = spoof_detector.check_liveness(img, face["bbox"])
                if not is_real:
                    print(f"Liveness: {liveness_score:.2f} (SPOOF DETECTED)")
                    print(f"Person: None")
                    print(f"Similarity: 0.00")
                    print(f"Status: SPOOF_DETECTED")
                    continue
                else:
                    print(f"Liveness: {liveness_score:.2f} (REAL)")

            with torch.no_grad():
                emb = model(face["tensor"])
            person, sim, status = match_face_embedding(emb, db, eff_threshold)
            print(f"Person: {person}")
            print(f"Similarity: {sim:.2f}")
            print(f"Status: {status}")
    else:
        # Default: Primary (largest/dominant) face
        primary = faces[0]
        if spoof_detector is not None:
            is_real, liveness_score, _ = spoof_detector.check_liveness(img, primary["bbox"])
            if not is_real:
                print(f"Liveness: {liveness_score:.2f} (SPOOF DETECTED)")
                print(f"Person: None")
                print(f"Similarity: 0.00")
                print(f"Status: SPOOF_DETECTED")
                return
            else:
                print(f"Liveness: {liveness_score:.2f} (REAL)")

        with torch.no_grad():
            emb = model(primary["tensor"])
        person, sim, status = match_face_embedding(emb, db, eff_threshold)
        print(f"Person: {person}")
        print(f"Similarity: {sim:.2f}")
        print(f"Status: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict identity of faces in an image.")
    parser.add_argument("image", type=str, help="Path to input image file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for MATCH (default from db: 0.45).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to face database (.pt).",
    )
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="Output predictions for all detected faces in the image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "xpu", "cpu", "cuda"],
        help="Compute device for ArcFace inference: 'auto', 'xpu', 'cpu', or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default="auto",
        choices=["auto", "openvino", "opencv"],
        help="Face detector backend: 'auto' (prefers OpenVINO GPU), 'openvino' (GPU), or 'opencv' (CPU).",
    )
    parser.add_argument(
        "--no-anti-spoof",
        action="store_true",
        help="Disable MiniFASNetV2 anti-spoofing liveness check.",
    )
    parser.add_argument(
        "--anti-spoof-threshold",
        type=float,
        default=DEFAULT_ANTI_SPOOF_THRESHOLD,
        help=f"Liveness threshold for MiniFASNetV2 (default: {DEFAULT_ANTI_SPOOF_THRESHOLD}).",
    )
    args = parser.parse_args()

    predict_image(
        image_path=Path(args.image),
        db_path=Path(args.db),
        threshold=args.threshold,
        all_faces=args.all_faces,
        device=args.device,
        detector_backend=args.detector_backend,
        anti_spoof=not args.no_anti_spoof,
        anti_spoof_threshold=args.anti_spoof_threshold,
    )
