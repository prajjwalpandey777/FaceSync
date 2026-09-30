"""Builds a robust face identity database from training images."""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import torch.nn.functional as F

from backbone.load_model import load_iresnet100
from face_detector import FaceDetector


DEFAULT_TRAIN_DIR = Path(r"C:\person\train")
DEFAULT_DB_PATH = PROJECT_ROOT / "weights" / "face_db.pt"


def enroll_identity(
    person_name: str,
    person_dir: Path,
    detector: FaceDetector,
    model: torch.nn.Module,
    device: torch.device,
    similarity_threshold: float = 0.40,
) -> Optional[Dict]:
    """Enroll all photos for a single identity while filtering background faces."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in person_dir.iterdir() if p.suffix.lower() in valid_extensions]
    )

    if not image_paths:
        print(f"[-] No valid images found in {person_dir}")
        return None

    print(f"\n[*] Processing identity: '{person_name}' ({len(image_paths)} images)")

    candidates = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] Failed to read: {img_path.name}")
            continue

        faces = detector.detect_faces(img, device=device)
        if not faces:
            print(f"  [-] No face detected in: {img_path.name}")
            continue

        # Extract embeddings for candidate faces in this image
        for face in faces:
            # Skip tiny background artifacts
            if face["area"] < detector.min_face_area:
                continue

            with torch.no_grad():
                emb = model(face["tensor"])
                emb = F.normalize(emb, p=2, dim=1)

            candidates.append(
                {
                    "emb": emb,
                    "file": img_path.name,
                    "area": face["area"],
                    "score": face["score"],
                }
            )

    if not candidates:
        print(f"[-] No valid faces detected for identity '{person_name}'")
        return None

    # Stack all candidate embeddings: [M, 512]
    all_embs = torch.cat([c["emb"] for c in candidates], dim=0)

    # Perform clustering to identify the main identity and reject background / friend faces
    if len(candidates) > 1:
        # Pairwise cosine similarity: [M, M]
        sim_matrix = torch.mm(all_embs, all_embs.t())
        k = min(6, len(candidates))
        topk_sims = torch.topk(sim_matrix, k=k, dim=1).values
        # Mean similarity to nearest neighbors (excluding self at index 0)
        mean_topk = torch.mean(topk_sims[:, 1:], dim=1)

        # Select the exemplar face with highest mutual cluster similarity
        best_idx = torch.argmax(mean_topk).item()
        exemplar_emb = all_embs[best_idx : best_idx + 1]
    else:
        exemplar_emb = all_embs[0:1]

    # Calculate similarity of every candidate to the exemplar
    sims_to_exemplar = torch.mm(all_embs, exemplar_emb.t()).squeeze(1)

    # For each image file, select the candidate with the highest similarity to exemplar
    file_to_best_candidate = {}
    excluded_faces = 0

    for i, c in enumerate(candidates):
        sim = sims_to_exemplar[i].item()
        fname = c["file"]

        if sim >= similarity_threshold:
            if fname not in file_to_best_candidate or sim > file_to_best_candidate[fname]["sim"]:
                file_to_best_candidate[fname] = {
                    "emb": c["emb"],
                    "sim": sim,
                    "file": fname,
                    "area": c["area"],
                }
        else:
            excluded_faces += 1

    enrolled = list(file_to_best_candidate.values())
    if not enrolled:
        print(f"[-] No faces met the identity consistency threshold for '{person_name}'")
        return None

    enrolled_embs = torch.cat([e["emb"] for e in enrolled], dim=0)  # [N, 512]
    enrolled_files = [e["file"] for e in enrolled]
    similarities = [e["sim"] for e in enrolled]

    # Compute normalized identity centroid
    centroid = torch.mean(enrolled_embs, dim=0, keepdim=True)
    centroid = F.normalize(centroid, p=2, dim=1)

    print(
        f"  [+] Enrolled {len(enrolled)}/{len(image_paths)} images "
        f"(filtered {excluded_faces} background/non-identity faces)"
    )
    print(
        f"  [+] Intra-identity similarities: min={min(similarities):.2f}, "
        f"mean={sum(similarities)/len(similarities):.2f}, max={max(similarities):.2f}"
    )

    return {
        "name": person_name,
        "embeddings": enrolled_embs.cpu(),
        "centroid": centroid.cpu(),
        "files": enrolled_files,
        "count": len(enrolled),
    }


def build_database(
    train_dir: Path = DEFAULT_TRAIN_DIR,
    output_db_path: Path = DEFAULT_DB_PATH,
    threshold: float = 0.45,
) -> None:
    """Scan training directory, extract embeddings, and save database."""
    train_dir = Path(train_dir)
    output_db_path = Path(output_db_path)

    if not train_dir.is_dir():
        print(f"Error: Training directory not found: {train_dir}")
        sys.exit(1)

    output_db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ArcFace IResNet-100 model...")
    model = load_iresnet100()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    print("Initializing face detector...")
    detector = FaceDetector()

    person_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        print(f"Error: No subdirectories found in {train_dir}")
        sys.exit(1)

    print(f"Found {len(person_dirs)} person folders: {[d.name for d in person_dirs]}")

    db_records = {}
    for pdir in person_dirs:
        record = enroll_identity(
            person_name=pdir.name,
            person_dir=pdir,
            detector=detector,
            model=model,
            device=device,
        )
        if record is not None:
            db_records[pdir.name] = record

    if not db_records:
        print("Error: No identities could be enrolled. Database not created.")
        sys.exit(1)

    database_payload = {
        "identities": list(db_records.keys()),
        "persons": db_records,
        "default_threshold": threshold,
        "embedding_dim": 512,
    }

    torch.save(database_payload, output_db_path)
    print(f"\n=======================================================")
    print(f"Face database successfully built and saved to:")
    print(f"  {output_db_path}")
    print(f"Total enrolled identities: {len(db_records)}")
    for name, rec in db_records.items():
        print(f"  - {name}: {rec['count']} reference embeddings")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ArcFace face recognition database.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(DEFAULT_TRAIN_DIR),
        help="Path to training directory containing person subfolders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Output path for database file (.pt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Default verification cosine similarity threshold (default: 0.45).",
    )
    args = parser.parse_args()

    build_database(
        train_dir=Path(args.train_dir),
        output_db_path=Path(args.output),
        threshold=args.threshold,
    )
