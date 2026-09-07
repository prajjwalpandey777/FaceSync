"""
Thin wrapper around the `face_recognition` (dlib) library.

Two things happen here:
1. enroll  - given ONE clear photo of a student's face, compute a 128-d embedding
             to store in the database.
2. scan    - given a live camera frame (which may contain 0-3+ faces), find every
             face, compute its embedding, and compare each one against a class's
             enrolled students to find matches.
"""
from io import BytesIO
from typing import List, Optional, Tuple

import face_recognition
import numpy as np
from PIL import Image, ImageOps
from fastapi import HTTPException, status

MAX_IMAGE_DIMENSION = 1024  # enrollment photos: keep quality high, this only runs once per student
SCAN_IMAGE_DIMENSION = 480  # live scan frames: smaller = much faster face detection, run every couple seconds


def _load_image_from_bytes(image_bytes: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # respect phone camera orientation
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not read image")

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension))

    return np.array(img)


def encode_single_face(image_bytes: bytes) -> List[float]:
    """Used during student enrollment. Requires exactly one face in the photo."""
    image = _load_image_from_bytes(image_bytes)
    locations = face_recognition.face_locations(image, model="hog")

    if len(locations) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No face detected in the enrollment photo")
    if len(locations) > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Multiple faces detected — please use a photo with only the student's face")

    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return encodings[0].tolist()


def encode_all_faces(image_bytes: bytes, max_dimension: int = SCAN_IMAGE_DIMENSION) -> List[List[float]]:
    """Used during a live attendance scan frame. Returns embeddings for every face found."""
    image = _load_image_from_bytes(image_bytes, max_dimension=max_dimension)
    locations = face_recognition.face_locations(image, model="hog")
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return [e.tolist() for e in encodings]


def best_match(
    unknown_encoding: List[float],
    candidates: List[Tuple[int, List[float]]],
    tolerance: float,
) -> Optional[Tuple[int, float]]:
    """
    candidates: list of (student_id, known_encoding)
    Returns (student_id, confidence_percent) for the closest match within tolerance, or None.
    """
    if not candidates:
        return None

    unknown = np.array(unknown_encoding)
    known_ids = [c[0] for c in candidates]
    known_encodings = np.array([c[1] for c in candidates])

    distances = np.linalg.norm(known_encodings - unknown, axis=1)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance > tolerance:
        return None

    # Convert distance (0 = identical) to an approximate confidence percentage for the UI.
    confidence = max(0.0, (1 - best_distance / tolerance)) * 40 + 60  # scales into ~60-100%
    return known_ids[best_idx], round(min(confidence, 99.9), 1)
