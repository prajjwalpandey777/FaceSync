"""Real-Time Webcam Face Recognition Attendance System.

Leverages official ArcFace IResNet-100, YuNet face detector, IoU tracking,
and multi-frame temporal voting to achieve high-accuracy attendance logging at real-time FPS.
"""

import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from attendance_manager import AttendanceManager
from backbone.load_model import load_iresnet100
from face_detector import FaceDetector
from predict import load_database, match_face_embedding
from tracker import FaceTracker, TrackedFace
from anti_spoof import MiniFASNetV2AntiSpoof


DEFAULT_DB_PATH = PROJECT_ROOT / "weights" / "face_db.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "backbone.pth"
DEFAULT_THRESHOLD = 0.45
DEFAULT_ANTI_SPOOF_MODEL = PROJECT_ROOT / "weights" / "2.7_80x80_MiniFASNetV2.onnx"
DEFAULT_ANTI_SPOOF_THRESHOLD = 0.60


class FrameGrabber(threading.Thread):
    """Background daemon thread for non-blocking, continuous webcam frame capture.

    Implements a *latest-frame-wins* single-slot buffer:
      - The camera thread calls cap.read() as fast as the hardware allows.
      - Only the most recently captured frame is kept; earlier frames are
        silently overwritten, so stale frames never accumulate.
      - The processing/main thread reads the latest frame instantly (< 1 µs)
        without ever blocking on camera I/O.

    This decouples camera exposure time (~33 ms for 30 fps USB cameras) from
    the inference pipeline, so both can proceed in parallel.
    """

    def __init__(self, source: Union[int, str]) -> None:
        super().__init__(daemon=True)
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._seq: int = 0          # Monotonically increasing frame counter
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Thread entry-point
    # ------------------------------------------------------------------
    def run(self) -> None:
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
                self._seq += 1
            if not ret:
                time.sleep(0.001)   # Brief pause on camera error, then retry

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Return (ret, frame, seq) of the latest captured frame.

        This call never blocks; it returns whatever the camera thread stored
        most recently.  The caller should compare *seq* with the previously
        seen sequence number to detect whether a new frame has arrived.
        """
        with self._lock:
            return self._ret, self._frame, self._seq

    def stop(self) -> None:
        """Signal the camera thread to stop and release the capture device."""
        self._stopped = True
        self._cap.release()


def draw_corner_rect(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 15,
) -> None:
    """Draw a modern bounding box with highlighted corners."""
    x, y, w, h = bbox
    # Main rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Corners
    # Top-Left
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)


def draw_hud(
    img: np.ndarray,
    fps: float,
    det_ms: float,
    rec_ms: float,
    session_count: int,
    banner_msg: Optional[str] = None,
    anti_spoof_active: bool = False,
    mirrored: bool = False,
) -> None:
    """Render top status HUD and temporary attendance notification banner."""
    ih, iw = img.shape[:2]

    # Semi-transparent HUD header background
    hud_h = 42
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (iw, hud_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Metrics text
    fps_text = f"FPS: {fps:4.1f}"
    det_text = f"Det: {det_ms:4.1f}ms"
    rec_text = f"Rec: {rec_ms:4.1f}ms" if rec_ms > 0 else "Rec: idle"
    as_text = "Live: ON" if anti_spoof_active else "Live: OFF"
    as_color = (0, 255, 120) if anti_spoof_active else (160, 160, 160)
    mir_text = " [MIRROR]" if mirrored else ""
    att_text = f"Present: {session_count}"

    cv2.putText(img, fps_text, (12, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, det_text, (130, 27), cv2.FONT_HERSHEY_DUPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, rec_text, (245, 27), cv2.FONT_HERSHEY_DUPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, as_text + mir_text, (365, 27), cv2.FONT_HERSHEY_DUPLEX, 0.52, as_color, 1, cv2.LINE_AA)
    cv2.putText(img, att_text, (iw - 150, 27), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 215, 255), 1, cv2.LINE_AA)

    # Notification banner if attendance was recently marked
    if banner_msg is not None:
        bw, bh = 460, 44
        bx1 = (iw - bw) // 2
        by1 = hud_h + 12
        bx2 = bx1 + bw
        by2 = by1 + bh

        # Banner card
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 80, 20), -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(
            img,
            banner_msg,
            (bx1 + 18, by1 + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


class WebcamAttendanceApp:
    """Modular real-time face recognition attendance application."""

    def __init__(
        self,
        camera_source: int = 0,
        db_path: Path = DEFAULT_DB_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        threshold: float = DEFAULT_THRESHOLD,
        vote_window: int = 5,
        vote_threshold: int = 4,
        recognize_interval: int = 8,
        csv_dir: Path = PROJECT_ROOT / "attendance",
        device: Optional[str] = None,
        detector_backend: str = "auto",
        mirror: bool = True,
        anti_spoof: bool = True,
        anti_spoof_threshold: float = DEFAULT_ANTI_SPOOF_THRESHOLD,
    ) -> None:
        self.camera_source = camera_source
        self.threshold = threshold
        self.recognize_interval = recognize_interval
        self.mirror = mirror

        # 1. Load Enrolled Database
        print("[*] Loading enrolled face database...")
        self.db = load_database(db_path)
        print(f"  [+] Loaded {len(self.db['persons'])} identities: {self.db['identities']}")

        # 2. Load ArcFace IResNet-100 Model
        print("[*] Loading ArcFace IResNet-100 backbone...")
        self.model = load_iresnet100(checkpoint_path=checkpoint_path, device=device)
        self.device = next(self.model.parameters()).device
        print(f"  [+] Backbone loaded on: {self.device}")

        # Warm-up pass to initialize compute kernels
        dummy = torch.zeros(1, 3, 112, 112, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        if self.device.type == "xpu":
            torch.xpu.synchronize()
        print(f"  [+] Compute device initialized and warmed up.")

        # 3. Initialize Modular Detector & Tracker
        print("[*] Initializing face detector & tracker...")
        self.detector = FaceDetector(backend=detector_backend)
        print(f"  [+] Detector active backend: {self.detector.backend_name}")
        self.tracker = FaceTracker(
            min_iou=0.30,
            vote_window=vote_window,
            vote_threshold=vote_threshold,
        )

        # 4. Initialize Attendance Manager
        self.attendance_mgr = AttendanceManager(output_dir=csv_dir)

        # 5. Initialize MiniFASNetV2 Anti-Spoofing
        self.anti_spoof: Optional[MiniFASNetV2AntiSpoof] = None
        if anti_spoof:
            print("[*] Initializing MiniFASNetV2 Anti-Spoofing...")
            self.anti_spoof = MiniFASNetV2AntiSpoof(
                model_path=DEFAULT_ANTI_SPOOF_MODEL,
                threshold=anti_spoof_threshold,
            )
            if self.anti_spoof.net is not None:
                print(f"  [+] Anti-Spoofing active (threshold={anti_spoof_threshold})")

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Run detection, tracking, anti-spoofing, ArcFace recognition, and UI rendering on one frame."""
        t0 = time.perf_counter()

        # Step 1: Continuous Fast Face Detection
        t_det_start = time.perf_counter()
        detections = self.detector.detect_faces(frame, device=self.device)
        t_det_ms = (time.perf_counter() - t_det_start) * 1000.0

        # Step 2: Multi-Face Tracking
        tracked_faces: List[TrackedFace] = self.tracker.update(detections)

        # Step 2.5: Anti-Spoofing Liveness Evaluation
        for track in tracked_faces:
            if self.anti_spoof is not None and self.anti_spoof.net is not None:
                is_real, liveness_score, _ = self.anti_spoof.check_liveness(frame, track.bbox)
                track.is_real = is_real
                track.liveness_score = liveness_score
            else:
                track.is_real = True
                track.liveness_score = 1.0

        # Step 3: Targeted / Intermittent ArcFace Recognition (batched) - only for REAL faces!
        t_rec_ms = 0.0
        faces_to_recognize = [
            t for t in tracked_faces
            if getattr(t, "is_real", True) and t.needs_recognition(interval=self.recognize_interval)
        ]
        if faces_to_recognize:
            rec_start = time.perf_counter()

            # Batch all recognition tensors into a single XPU forward pass.
            batch_tensor = torch.cat(
                [t.tensor.to(self.device) for t in faces_to_recognize], dim=0
            )  # [N, 3, 112, 112]
            with torch.no_grad():
                batch_embs = self.model(batch_tensor)  # [N, 512]
            # Single CPU transfer: implicitly synchronizes XPU once for all faces
            batch_embs_cpu = batch_embs.cpu()

            for i, track in enumerate(faces_to_recognize):
                person_name, sim, status = match_face_embedding(
                    batch_embs_cpu[i : i + 1], self.db, threshold=self.threshold
                )
                track.add_prediction(person_name, sim, status)

                # Check if attendance can be confirmed and marked
                if (
                    track.is_confirmed
                    and track.confirmed_name != "Unknown"
                    and not track.attendance_marked
                ):
                    success, _ = self.attendance_mgr.mark_attendance(
                        track.confirmed_name, track.confirmed_similarity
                    )
                    if success:
                        track.attendance_marked = True

            t_rec_ms = (time.perf_counter() - rec_start) * 1000.0


        # Step 4: Render Bounding Boxes & Labels
        display_frame = frame.copy()
        for track in tracked_faces:
            x, y, w, h = track.bbox
            name = track.confirmed_name
            sim = track.confirmed_similarity
            status = track.confirmed_status
            is_real = getattr(track, "is_real", True)
            liveness = getattr(track, "liveness_score", 1.0)

            # Select color based on liveness, recognition & attendance state
            if not is_real:
                color = (0, 0, 230)  # Crimson: Spoof attack
                label_text = f"SPOOF DETECTED [{liveness:.2f}] REJECTED"
            elif status == "MATCH" and (track.attendance_marked or self.attendance_mgr.is_already_marked(name)):
                color = (0, 220, 0)  # Green: Confirmed & Marked
                label_text = f"{name.upper()} [{sim:.2f}] MATCH (Live: {liveness:.2f})"
            elif status == "CONFIRMING":
                color = (0, 215, 255)  # Cyan/Yellow: Confirming votes
                curr_votes, req_votes = track.confirmation_progress
                label_text = f"{name} [{sim:.2f}] Confirming ({curr_votes}/{req_votes})"
            else:
                color = (50, 50, 240)  # Red: Unknown / Non-enrolled
                label_text = f"Unknown [{sim:.2f}] UNKNOWN (Live: {liveness:.2f})"

            # Draw box with corner accents
            draw_corner_rect(display_frame, (x, y, w, h), color, thickness=2)

            # Draw label badge above bbox
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            badge_w = label_size[0] + 12
            badge_h = label_size[1] + 10
            badge_y1 = max(0, y - badge_h)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = x + badge_w

            cv2.rectangle(display_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            text_color = (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255)
            cv2.putText(
                display_frame,
                label_text,
                (badge_x1 + 6, badge_y2 - 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.50,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return display_frame, t_det_ms, t_rec_ms

    def run(
        self,
        max_frames: Optional[int] = None,
        headless: bool = False,
        async_capture: bool = True,
    ) -> Dict[str, float]:
        """Start the webcam capture loop and display attendance feed.

        Args:
            max_frames:     Stop after this many *processed* frames (for benchmarking).
            headless:       Suppress the GUI display window.
            async_capture:  When True (default), camera capture runs on a background
                            thread so the inference pipeline never blocks on I/O.
                            When False, reverts to the previous synchronous behaviour.
        """
        print(f"\n[*] Opening camera source: {self.camera_source}")

        window_name = "Real-Time Face Recognition Attendance (ArcFace IResNet-100)"

        if async_capture:
            # -----------------------------------------------------------------
            # Async path: FrameGrabber runs cap.read() on a background daemon
            # thread.  The main thread grabs the latest available frame
            # instantly (non-blocking) and processes at its own pace.
            # -----------------------------------------------------------------
            grabber = FrameGrabber(self.camera_source)
            if not grabber.is_opened():
                print(f"Error: Could not open camera {self.camera_source}")
                sys.exit(1)
            grabber.start()
            print("[+] Async camera capture thread started.")
            # Give the camera a moment to deliver its first frame
            time.sleep(0.3)
        else:
            # -----------------------------------------------------------------
            # Sync path: original blocking cap.read() behaviour (kept as
            # --sync-capture fallback for direct comparison).
            # -----------------------------------------------------------------
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                print(f"Error: Could not open camera {self.camera_source}")
                sys.exit(1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("[+] Camera opened (synchronous capture).")

        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("[*] Press 'q' or 'ESC' to exit.")
        print("[*] Press 'c' to clear session attendance.")
        print("[*] Press 's' to save a screenshot.\n")

        # Metrics
        frame_times: List[float] = []      # Per-frame pipeline time (excl. camera wait)
        det_times:   List[float] = []
        rec_times:   List[float] = []
        frame_idx = 0
        fps = 0.0
        display_frame: Optional[np.ndarray] = None
        last_seq = -1                       # Last camera-sequence number we processed
        wall_start = time.perf_counter()    # For real-world (wall-clock) FPS

        try:
            while True:
                # ----------------------------------------------------------------
                # Frame acquisition
                # ----------------------------------------------------------------
                if async_capture:
                    ret, frame, seq = grabber.read()

                    if not ret or frame is None or seq == last_seq:
                        # No new frame yet — keep display fresh and handle input
                        if not headless and display_frame is not None:
                            cv2.imshow(window_name, display_frame)
                        key = (cv2.waitKey(1) & 0xFF) if not headless else 0xFF
                        if key in (ord("q"), 27):
                            print("[*] Exit key pressed.")
                            break
                        elif key == ord("c"):
                            self.attendance_mgr.clear_session()
                        elif key == ord("s") and display_frame is not None:
                            ss_path = PROJECT_ROOT / f"screenshot_{int(time.time())}.jpg"
                            cv2.imwrite(str(ss_path), display_frame)
                            print(f"[+] Screenshot saved to: {ss_path}")
                        if headless:
                            time.sleep(0.001)   # Avoid busy-spinning in headless mode
                        continue             # Do not count non-new frames

                    last_seq = seq          # Mark this sequence as processed

                else:
                    # Synchronous blocking read
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[!] End of video stream or failed to grab frame.")
                        break

                # Apply mirror horizontal flip for natural webcam experience
                if self.mirror and frame is not None:
                    frame = cv2.flip(frame, 1)

                # ----------------------------------------------------------------
                # Process new frame (detection → tracking → recognition → render)
                # ----------------------------------------------------------------
                t_frame_start = time.perf_counter()
                display_frame, det_ms, rec_ms = self.process_frame(frame)
                t_frame_end = time.perf_counter()

                frame_duration = t_frame_end - t_frame_start
                frame_times.append(frame_duration)
                det_times.append(det_ms)
                if rec_ms > 0:
                    rec_times.append(rec_ms)

                # Rolling pipeline FPS (processing speed, not camera-bound)
                recent_window = frame_times[-30:]
                fps = len(recent_window) / sum(recent_window) if recent_window else 0.0

                # Render HUD overlay
                banner_msg = self.attendance_mgr.get_banner_message()
                draw_hud(
                    display_frame,
                    fps=fps,
                    det_ms=det_ms,
                    rec_ms=rec_ms,
                    session_count=self.attendance_mgr.get_session_count(),
                    banner_msg=banner_msg,
                    anti_spoof_active=(self.anti_spoof is not None and self.anti_spoof.net is not None),
                    mirrored=self.mirror,
                )

                frame_idx += 1

                if not headless:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[*] Exit key pressed.")
                        break
                    elif key == ord("c"):
                        self.attendance_mgr.clear_session()
                    elif key == ord("s"):
                        ss_path = PROJECT_ROOT / f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(str(ss_path), display_frame)
                        print(f"[+] Screenshot saved to: {ss_path}")

                if max_frames is not None and frame_idx >= max_frames:
                    print(f"[*] Reached benchmark limit of {max_frames} frames.")
                    break

        finally:
            if async_capture:
                grabber.stop()
            else:
                cap.release()
            if not headless:
                cv2.destroyAllWindows()

        wall_elapsed = time.perf_counter() - wall_start
        wall_fps = frame_idx / wall_elapsed if wall_elapsed > 0 else 0.0
        pipeline_fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
        avg_det_ms = float(np.mean(det_times)) if det_times else 0.0
        avg_rec_ms = float(np.mean(rec_times)) if rec_times else 0.0

        metrics = {
            "total_frames":    frame_idx,
            "wall_fps":        wall_fps,
            "pipeline_fps":    pipeline_fps,
            "avg_det_ms":      avg_det_ms,
            "avg_rec_ms":      avg_rec_ms,
            "enrolled_marked": self.attendance_mgr.get_session_count(),
        }

        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Processed Frames     : {frame_idx}")
        print(f"  Capture Mode         : {'Async (background thread)' if async_capture else 'Sync (blocking)'}")
        print(f"  Detector Backend     : {self.detector.backend_name}")
        print(f"  ArcFace Device       : {self.device}")
        print(f"  Wall-Clock FPS       : {wall_fps:.2f} FPS  (real-world throughput)")
        print(f"  Pipeline FPS         : {pipeline_fps:.2f} FPS  (processing speed excl. camera wait)")
        print(f"  Face Detector Avg    : {avg_det_ms:.2f} ms")
        print(f"  ArcFace Avg (active) : {avg_rec_ms:.2f} ms")
        print(f"  Session Attendance   : {metrics['enrolled_marked']} identities marked")
        print("=" * 60 + "\n")

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Webcam Face Recognition Attendance.")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera device index (default: 0) or path to a video file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for MATCH (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--vote-window",
        type=int,
        default=5,
        help="Sliding window size for multi-frame temporal voting (default: 5).",
    )
    parser.add_argument(
        "--vote-threshold",
        type=int,
        default=4,
        help="Required matching votes within window to confirm identity (default: 4).",
    )
    parser.add_argument(
        "--recognize-interval",
        type=int,
        default=8,
        help="Frame interval to run ArcFace on tracked faces once confirmed (default: 8).",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(PROJECT_ROOT / "attendance"),
        help="Directory to save attendance CSV logs (default: attendance).",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=None,
        help="Run for N frames then exit and report metrics (for automated testing).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI display window (for background/automated testing).",
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
        "--sync-capture",
        action="store_true",
        help="Use synchronous (blocking) camera capture instead of the default async background thread.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror flip (default: mirrored preview for natural webcam experience).",
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

    # Determine if camera argument is an int device index or video file string
    try:
        cam_src = int(args.camera)
    except ValueError:
        cam_src = args.camera

    app = WebcamAttendanceApp(
        camera_source=cam_src,
        threshold=args.threshold,
        vote_window=args.vote_window,
        vote_threshold=args.vote_threshold,
        recognize_interval=args.recognize_interval,
        csv_dir=Path(args.csv_dir),
        device=args.device,
        detector_backend=args.detector_backend,
        mirror=not args.no_mirror,
        anti_spoof=not args.no_anti_spoof,
        anti_spoof_threshold=args.anti_spoof_threshold,
    )

    app.run(
        max_frames=args.benchmark_frames,
        headless=args.headless,
        async_capture=not args.sync_capture,
    )

