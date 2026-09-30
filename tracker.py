"""Multi-face tracker with temporal voting and track management."""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class TrackedFace:
    """Represents a single continuous face track across frames."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray],
        tensor: torch.Tensor,
        aligned_bgr: np.ndarray,
        vote_window: int = 5,
        vote_threshold: int = 4,
    ) -> None:
        self.track_id = track_id
        self.bbox = bbox
        self.landmarks = landmarks
        self.tensor = tensor
        self.aligned_bgr = aligned_bgr
        self.vote_window = vote_window
        self.vote_threshold = vote_threshold

        # Prediction history: deque of (name, similarity, status)
        self.history = deque(maxlen=vote_window)

        # Recognition & confirmation state
        self.confirmed_name: str = "Unknown"
        self.confirmed_similarity: float = 0.0
        self.confirmed_status: str = "UNKNOWN"
        self.is_confirmed: bool = False
        self.attendance_marked: bool = False

        # Timing / frequency counters
        self.frames_since_recognition: int = 0
        self.frames_unseen: int = 0

    def update_detection(
        self,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray],
        tensor: torch.Tensor,
        aligned_bgr: np.ndarray,
    ) -> None:
        """Update spatial location and aligned face representation from current frame."""
        self.bbox = bbox
        self.landmarks = landmarks
        self.tensor = tensor
        self.aligned_bgr = aligned_bgr
        self.frames_unseen = 0
        self.frames_since_recognition += 1

    def add_prediction(self, name: str, similarity: float, status: str) -> None:
        """Record a recognition result and update temporal voting status."""
        self.history.append((name, similarity, status))
        self.frames_since_recognition = 0

        # Tally votes across sliding window
        votes: Dict[str, List[float]] = {}
        for h_name, h_sim, _ in self.history:
            votes.setdefault(h_name, []).append(h_sim)

        # Find the candidate identity with the most votes
        top_name, sim_list = max(votes.items(), key=lambda item: len(item[1]))
        vote_count = len(sim_list)
        avg_sim = float(np.mean(sim_list))

        if vote_count >= self.vote_threshold:
            self.confirmed_name = top_name
            self.confirmed_similarity = avg_sim
            self.confirmed_status = "MATCH" if top_name != "Unknown" else "UNKNOWN"
            self.is_confirmed = True
        else:
            # Still in confirmation phase
            self.confirmed_name = top_name
            self.confirmed_similarity = avg_sim
            self.confirmed_status = "CONFIRMING"
            self.is_confirmed = False

    def needs_recognition(self, interval: int = 8) -> bool:
        """Determine if this track should be evaluated by the heavy ArcFace backbone."""
        # If not yet confirmed, evaluate every frame to build voting buffer quickly
        if not self.is_confirmed:
            return True
        # Once confirmed, re-evaluate only periodically to maintain identity
        return self.frames_since_recognition >= interval

    @property
    def confirmation_progress(self) -> Tuple[int, int]:
        """Return (current_matching_votes, required_votes)."""
        if not self.history:
            return (0, self.vote_threshold)
        votes = [1 for h_name, _, _ in self.history if h_name == self.confirmed_name]
        return (len(votes), self.vote_threshold)


class FaceTracker:
    """Associates detected bounding boxes across frames to maintain stable tracks."""

    def __init__(
        self,
        min_iou: float = 0.30,
        max_unseen_frames: int = 15,
        vote_window: int = 5,
        vote_threshold: int = 4,
    ) -> None:
        self.min_iou = min_iou
        self.max_unseen_frames = max_unseen_frames
        self.vote_window = vote_window
        self.vote_threshold = vote_threshold
        self.next_track_id: int = 1
        self.tracks: List[TrackedFace] = []

    def update(self, detected_faces: List[Dict]) -> List[TrackedFace]:
        """Update tracker with detections from current frame.

        Args:
            detected_faces: List of dicts returned by FaceDetector.detect_faces()

        Returns:
            List of active TrackedFace instances currently visible in this frame.
        """
        if not detected_faces:
            # Mark all existing tracks as unseen
            for track in self.tracks:
                track.frames_unseen += 1
                track.frames_since_recognition += 1
            # Prune dead tracks
            self.tracks = [t for t in self.tracks if t.frames_unseen <= self.max_unseen_frames]
            return []

        if not self.tracks:
            # Initialize tracks for all detected faces
            for det in detected_faces:
                track = TrackedFace(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    landmarks=det.get("landmarks"),
                    tensor=det["tensor"],
                    aligned_bgr=det["aligned_bgr"],
                    vote_window=self.vote_window,
                    vote_threshold=self.vote_threshold,
                )
                self.next_track_id += 1
                self.tracks.append(track)
            return self.tracks

        # Compute IoU matrix between existing tracks and new detections
        num_tracks = len(self.tracks)
        num_dets = len(detected_faces)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detected_faces):
                iou_matrix[t_idx, d_idx] = calculate_iou(track.bbox, det["bbox"])

        matched_tracks = set()
        matched_dets = set()

        # Greedy match based on highest IoU
        while True:
            if iou_matrix.size == 0:
                break
            max_val = np.max(iou_matrix)
            if max_val < self.min_iou:
                break
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track = self.tracks[t_idx]
            det = detected_faces[d_idx]

            track.update_detection(
                bbox=det["bbox"],
                landmarks=det.get("landmarks"),
                tensor=det["tensor"],
                aligned_bgr=det["aligned_bgr"],
            )
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

            # Invalidate this row and column
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        # For unmatched existing tracks, mark as unseen
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.frames_unseen += 1
                track.frames_since_recognition += 1

        # For unmatched detections, create new tracks
        for d_idx, det in enumerate(detected_faces):
            if d_idx not in matched_dets:
                new_track = TrackedFace(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    landmarks=det.get("landmarks"),
                    tensor=det["tensor"],
                    aligned_bgr=det["aligned_bgr"],
                    vote_window=self.vote_window,
                    vote_threshold=self.vote_threshold,
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # Prune expired tracks
        self.tracks = [t for t in self.tracks if t.frames_unseen <= self.max_unseen_frames]

        # Return only currently visible tracks (frames_unseen == 0)
        return [t for t in self.tracks if t.frames_unseen == 0]
