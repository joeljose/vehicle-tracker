#!/usr/bin/env python3
"""Draw ROI polygon and entry/exit lines on a video frame, save to site config.

Usage:
    python3 tools/draw_config.py <video_path> <site_id>

Example:
    python3 tools/draw_config.py data_collection/test_clips/lytle_south_10s.mp4 lytle_south

Controls:
    Mode 1 — ROI Polygon (starts here):
        Left-click   Add vertex
        Right-click  Undo last vertex
        Enter        Confirm polygon, switch to Line mode
        r            Reset polygon

    Mode 2 — Entry/Exit Lines:
        Left-click   Set line start (1st click) then end (2nd click)
        Right-click  Cancel current line / undo
        After 2nd click, type label with keyboard, press Enter to confirm
        d            Delete last completed line

    General:
        s            Save config and exit
        q / Esc      Quit without saving
        Space        Toggle motion heatmap overlay
        [ / ]        Seek backward/forward 2 seconds
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SITES_DIR = Path(__file__).parent.parent / "backend" / "config" / "sites"
WINDOW = "Draw Config"

# Colors (BGR)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 120, 0)
RED = (0, 0, 255)
BLUE = (255, 100, 50)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)

LINE_COLORS = [RED, BLUE, CYAN, MAGENTA, ORANGE, YELLOW]


class DrawConfig:
    def __init__(self, video_path: str, site_id: str):
        self.video_path = video_path
        self.site_id = site_id
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open {video_path}")
            sys.exit(1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        ret, self.frame = self.cap.read()
        if not ret:
            print("ERROR: Cannot read first frame")
            sys.exit(1)

        self.h, self.w = self.frame.shape[:2]
        self.motion_overlay = None
        self.show_motion = False

        # State machine: roi -> line -> label -> line -> ...
        self.mode = "roi"
        self.roi_points: list[tuple[int, int]] = []
        self.roi_confirmed = False

        self.lines: list[dict] = []
        self.current_line_start: tuple[int, int] | None = None
        self.pending_line: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.label_text = ""  # text being typed for line label
        self.mouse_pos = (0, 0)

        self._load_existing()

    def _load_existing(self):
        path = SITES_DIR / f"{self.site_id}.json"
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        has_roi = bool(data.get("roi_polygon"))
        has_lines = bool(data.get("entry_exit_lines"))

        if not has_roi and not has_lines:
            return

        # Show what exists and ask what to do
        print(f"\nExisting config found for '{self.site_id}':")
        if has_roi:
            print(f"  ROI: {len(data['roi_polygon'])} vertices")
        if has_lines:
            for arm_id, ld in data["entry_exit_lines"].items():
                print(f"  Line: {arm_id} \"{ld['label']}\"")

        print(f"\nOptions:")
        print(f"  k = Keep everything (add more lines on top)")
        print(f"  l = Keep ROI, clear lines (redraw lines only)")
        print(f"  n = Start fresh (clear everything)")
        print(f"  q = Quit")

        while True:
            choice = input("\nChoice [k/l/n/q]: ").strip().lower()
            if choice in ("k", "l", "n", "q"):
                break
            print("  Invalid choice.")

        if choice == "q":
            sys.exit(0)

        if choice == "n":
            print("Starting fresh.")
            return

        if has_roi:
            self.roi_points = [tuple(p) for p in data["roi_polygon"]]
            self.roi_confirmed = True
            self.mode = "line"
            print(f"Loaded ROI ({len(self.roi_points)} vertices)")

        if choice == "k" and has_lines:
            for arm_id, line_data in data["entry_exit_lines"].items():
                self.lines.append({
                    "arm_id": arm_id,
                    "label": line_data["label"],
                    "start": tuple(line_data["start"]),
                    "end": tuple(line_data["end"]),
                })
            print(f"Loaded {len(self.lines)} lines")
        elif choice == "l":
            print("Lines cleared. Draw new ones.")

    def _seek(self, offset_sec: float):
        new_idx = self.current_frame_idx + int(offset_sec * self.fps)
        new_idx = max(0, min(new_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.current_frame_idx = new_idx

    def _compute_motion(self):
        print("Computing motion heatmap...")
        cap = cv2.VideoCapture(self.video_path)
        motion_map = np.zeros((self.h, self.w), dtype=np.float64)

        ret, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        count = 0
        skip = max(1, int(self.fps / 10))
        max_frames = min(self.total_frames, int(60 * self.fps))

        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % skip != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            motion_map += diff.astype(np.float64)
            prev_gray = gray

        cap.release()

        if motion_map.max() > 0:
            p99 = np.percentile(motion_map[motion_map > 0], 99)
            clipped = np.clip(motion_map, 0, p99)
            norm = (clipped / p99 * 255).astype(np.uint8)
        else:
            norm = np.zeros((self.h, self.w), dtype=np.uint8)

        self.motion_overlay = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        print("Motion heatmap ready. Press Space to toggle.")

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "roi" and not self.roi_confirmed:
                self.roi_points.append((x, y))
            elif self.mode == "line":
                if self.current_line_start is None:
                    self.current_line_start = (x, y)
                else:
                    self.pending_line = (self.current_line_start, (x, y))
                    self.current_line_start = None
                    self.label_text = ""
                    self.mode = "label"

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "roi" and not self.roi_confirmed and self.roi_points:
                self.roi_points.pop()
            elif self.mode == "line":
                self.current_line_start = None
            elif self.mode == "label":
                self.pending_line = None
                self.label_text = ""
                self.mode = "line"

    def _confirm_line(self):
        """Confirm the pending line with the typed label."""
        if not self.pending_line:
            return

        start, end = self.pending_line
        label = self.label_text.strip()
        if not label:
            label = f"line_{len(self.lines) + 1}"

        # Use label as arm_id (lowercase, replace spaces with underscores)
        arm_id = label.lower().replace(" ", "_").replace("-", "_")

        self.lines.append({
            "arm_id": arm_id,
            "label": label,
            "start": start,
            "end": end,
        })
        print(f"  Added: {arm_id} \"{label}\" ({start[0]},{start[1]})->({end[0]},{end[1]})")

        self.pending_line = None
        self.label_text = ""
        self.mode = "line"

    def _cancel_line(self):
        self.pending_line = None
        self.label_text = ""
        self.mode = "line"

    def _draw(self):
        if self.show_motion and self.motion_overlay is not None:
            canvas = cv2.addWeighted(self.frame, 0.5, self.motion_overlay, 0.5, 0)
        else:
            canvas = self.frame.copy()

        # ROI
        if len(self.roi_points) > 0:
            pts = np.array(self.roi_points, np.int32)
            if self.roi_confirmed:
                cv2.polylines(canvas, [pts], True, GREEN, 2)
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (0, 60, 0))
                canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
            else:
                cv2.polylines(canvas, [pts], False, GREEN, 2)
                for p in self.roi_points:
                    cv2.circle(canvas, p, 5, GREEN, -1)
                if self.roi_points:
                    cv2.line(canvas, self.roi_points[-1], self.mouse_pos, DARK_GREEN, 1)
                    # Ghost closing line
                    if len(self.roi_points) >= 3:
                        cv2.line(canvas, self.roi_points[0], self.mouse_pos, DARK_GREEN, 1, cv2.LINE_AA)

        # Completed lines
        for i, line in enumerate(self.lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            cv2.line(canvas, line["start"], line["end"], color, 3)
            mid_x = (line["start"][0] + line["end"][0]) // 2
            mid_y = (line["start"][1] + line["end"][1]) // 2
            cv2.putText(canvas, line["label"], (mid_x + 8, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(canvas, line["start"], 6, color, -1)
            cv2.circle(canvas, line["end"], 6, WHITE, 2)

        # Pending line (waiting for label)
        if self.pending_line:
            s, e = self.pending_line
            cv2.line(canvas, s, e, YELLOW, 3)
            cv2.circle(canvas, s, 6, YELLOW, -1)
            cv2.circle(canvas, e, 6, WHITE, 2)
            # Draw label input box near the line midpoint
            mid_x = (s[0] + e[0]) // 2
            mid_y = (s[1] + e[1]) // 2
            display_text = self.label_text + "_"
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            box_w = max(text_size[0] + 20, 200)
            box_x = min(mid_x, self.w - box_w - 10)
            box_y = max(mid_y - 40, 10)
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + 35), BLACK, -1)
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + 35), YELLOW, 2)
            cv2.putText(canvas, display_text, (box_x + 10, box_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

        # Line in progress (start set, dragging to end)
        elif self.current_line_start is not None:
            cv2.line(canvas, self.current_line_start, self.mouse_pos, YELLOW, 2)
            cv2.circle(canvas, self.current_line_start, 6, YELLOW, -1)

        # Cursor coordinates
        cx, cy = self.mouse_pos
        if self.mode != "label":
            cv2.putText(canvas, f"({cx},{cy})", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

        # Bottom HUD
        mode_names = {"roi": "ROI POLYGON", "line": "LINES", "label": "TYPE LABEL"}
        time_sec = self.current_frame_idx / self.fps if self.fps > 0 else 0
        info = f"{mode_names[self.mode]} | {self.site_id} | t={time_sec:.1f}s | Lines: {len(self.lines)}"
        cv2.rectangle(canvas, (0, self.h - 35), (self.w, self.h), BLACK, -1)
        cv2.putText(canvas, info, (10, self.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Top hint
        if self.mode == "roi" and not self.roi_confirmed:
            hint = "Click=add vertex | Right-click=undo | Enter=confirm | r=reset"
        elif self.mode == "line":
            if self.current_line_start is None:
                hint = "Click=start line | d=del last | c=clear all | s=save | q=quit | Space=heatmap"
            else:
                hint = "Click=set endpoint | Right-click=cancel"
        elif self.mode == "label":
            hint = "Type label, Enter=confirm | Esc=cancel | Backspace=delete"
        else:
            hint = ""
        cv2.rectangle(canvas, (0, 0), (self.w, 35), (0, 0, 0, 128), -1)
        cv2.putText(canvas, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        cv2.imshow(WINDOW, canvas)

    def save(self):
        SITES_DIR.mkdir(parents=True, exist_ok=True)
        path = SITES_DIR / f"{self.site_id}.json"

        data = {
            "site_id": self.site_id,
            "roi_polygon": [list(p) for p in self.roi_points],
            "entry_exit_lines": {},
        }
        for line in self.lines:
            data["entry_exit_lines"][line["arm_id"]] = {
                "label": line["label"],
                "start": list(line["start"]),
                "end": list(line["end"]),
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved to {path}")
        print(f"  ROI: {len(self.roi_points)} vertices")
        print(f"  Lines: {len(self.lines)}")
        for line in self.lines:
            print(f"    {line['arm_id']}: \"{line['label']}\"")

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(self.w, 1600), min(self.h, 900))
        cv2.setMouseCallback(WINDOW, self._mouse_callback)

        print(f"\nDrawing config for site '{self.site_id}'")
        print(f"Video: {self.video_path} ({self.w}x{self.h}, {self.fps:.0f}fps)")
        print(f"\n1. Click to draw ROI polygon, press Enter to confirm")
        print(f"2. Click start+end of each line, type label, press Enter")
        print(f"3. Press 's' to save, 'q' to quit\n")

        while True:
            self._draw()
            key = cv2.waitKey(30) & 0xFF

            if key == 255:  # no key
                continue

            # === LABEL MODE: all keys go to text input ===
            if self.mode == "label":
                if key == 13:  # Enter — confirm label
                    self._confirm_line()
                elif key == 27:  # Esc — cancel
                    self._cancel_line()
                elif key == 8 or key == 127:  # Backspace / Delete
                    self.label_text = self.label_text[:-1]
                elif 32 <= key <= 126:  # Printable ASCII
                    self.label_text += chr(key)
                continue

            # === ROI / LINE MODE ===
            if key == ord("q") or key == 27:  # q or Esc
                print("Quit without saving.")
                break

            elif key == ord("s"):
                self.save()
                break

            elif key == 13:  # Enter
                if self.mode == "roi" and not self.roi_confirmed:
                    if len(self.roi_points) >= 3:
                        self.roi_confirmed = True
                        self.mode = "line"
                        print(f"ROI confirmed ({len(self.roi_points)} vertices)")
                    else:
                        print("Need at least 3 points for ROI.")

            elif key == ord("r"):
                if self.mode == "roi" and not self.roi_confirmed:
                    self.roi_points.clear()
                    print("ROI reset.")
                elif self.mode == "line":
                    self.current_line_start = None

            elif key == ord(" "):
                if self.motion_overlay is None:
                    self._compute_motion()
                self.show_motion = not self.show_motion

            elif key == ord("["):
                self._seek(-2)
            elif key == ord("]"):
                self._seek(2)

            elif key == ord("d"):
                if self.mode == "line" and self.lines:
                    removed = self.lines.pop()
                    print(f"Deleted: {removed['label']}")

            elif key == ord("c"):
                if self.mode == "line" and self.lines:
                    self.lines.clear()
                    print("All lines cleared.")

        cv2.destroyAllWindows()
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Draw ROI and entry/exit lines on a video frame")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("site_id", help="Site identifier (e.g. 741_73, lytle_south)")
    args = parser.parse_args()

    tool = DrawConfig(args.video_path, args.site_id)
    tool.run()


if __name__ == "__main__":
    main()
