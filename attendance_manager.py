"""Attendance management, session deduplication, and CSV logging."""

import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple


DEFAULT_ATTENDANCE_DIR = Path(__file__).resolve().parent / "attendance"


class AttendanceManager:
    """Manages attendance records, prevents duplicates, and logs to daily CSV files."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_ATTENDANCE_DIR,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds

        # In-memory session registry: name -> {"timestamp": float, "date_str": str, "time_str": str, "similarity": float}
        self.session_records: Dict[str, Dict] = {}

        # Recent banner notification state: (text, expiry_time)
        self.active_banner: Optional[Tuple[str, float]] = None

    def _get_daily_csv_path(self) -> Path:
        """Get file path for today's attendance CSV log."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        return self.output_dir / f"attendance_{today_str}.csv"

    def is_already_marked(self, person_name: str) -> bool:
        """Check if person has already been marked in the current session / cooldown period."""
        if person_name not in self.session_records:
            return False

        if self.cooldown_seconds is None:
            # Session-wide single entry: already marked
            return True

        # Cooldown check
        last_marked_time = self.session_records[person_name]["timestamp"]
        elapsed = time.time() - last_marked_time
        return elapsed < self.cooldown_seconds

    def mark_attendance(
        self, person_name: str, similarity: float
    ) -> Tuple[bool, str]:
        """Record attendance for a confirmed person.

        Args:
            person_name: Enrolled identity name (must not be 'Unknown')
            similarity: Cosine similarity score

        Returns:
            (success, message)
        """
        if person_name in ("Unknown", "None", ""):
            return False, "Cannot mark attendance for Unknown person"

        now = datetime.now()
        timestamp = time.time()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if self.is_already_marked(person_name):
            prev_time = self.session_records[person_name]["time_str"]
            return False, f"Attendance already marked for {person_name} at {prev_time}"

        # Register in session memory
        record = {
            "timestamp": timestamp,
            "date_str": date_str,
            "time_str": time_str,
            "similarity": similarity,
        }
        self.session_records[person_name] = record

        # Append to daily CSV
        csv_path = self._get_daily_csv_path()
        file_exists = csv_path.is_file()

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Date", "Time", "Person", "Similarity", "Status"])
            writer.writerow([date_str, time_str, person_name, f"{similarity:.4f}", "PRESENT"])

        # Set active on-screen notification (displays for 3.5 seconds)
        banner_msg = f"Attendance Marked: {person_name.upper()} at {time_str}"
        self.active_banner = (banner_msg, timestamp + 3.5)

        print(f"[ATTENDANCE RECORDED] {date_str} {time_str} | {person_name} (Similarity: {similarity:.2f})")
        return True, banner_msg

    def get_banner_message(self) -> Optional[str]:
        """Return currently active on-screen banner message, or None if expired."""
        if self.active_banner is None:
            return None
        msg, expiry = self.active_banner
        if time.time() > expiry:
            self.active_banner = None
            return None
        return msg

    def get_session_count(self) -> int:
        """Return number of unique identities marked during this session."""
        return len(self.session_records)

    def clear_session(self) -> None:
        """Reset session records."""
        self.session_records.clear()
        self.active_banner = None
        print("[*] Attendance session cleared.")
