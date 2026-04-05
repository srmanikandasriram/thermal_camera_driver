"""
FLIR Lepton Thermal Camera Wrapper

This module provides a wrapper class for FLIR Lepton thermal cameras with
background frame capture, telemetry parsing, and logging capabilities,
mirroring the interface of BosonWithTelemetry for consistent use across
recording scripts.

When run directly, launches a full-featured live viewer with an on-screen
HUD showing filter states, temperatures, and keyboard shortcuts to toggle
camera filters and display settings.

Author: [Your Name]
Date: March 2026
"""

import time
import logging
import threading
import numpy as np
from typing import Optional, Tuple, Dict

from flirpy.camera.lepton import Lepton


class LeptonWrapper:
    """
    FLIR Lepton camera wrapper with background capture and logging support.

    Manages a background thread that continuously calls grab() on the Lepton,
    providing the same logged_images / logged_tstamps interface as
    BosonWithTelemetry so recording scripts can treat both cameras uniformly.

    Attributes:
        logged_images (list): Raw frames WITH telemetry footer rows
        logged_tstamps (list): System timestamps (time.time()) per frame
        logged_cam_tstamps (list): Camera-internal timestamps (uptime-based)
        logged_frame_numbers (list): Camera frame counter per frame
        enable_logging (bool): Flag to control data logging
        timestamp_offset (float): system_time − camera_time (computed on first frame)
    """

    def __init__(self, device: Optional[int] = None,
                 loglevel: int = logging.WARNING) -> None:
        """
        Initialize the Lepton camera and start the background capture thread.

        Args:
            device: Camera device index (default: None for auto-detection)
            loglevel: Logging level (default: logging.WARNING)
        """
        logging.basicConfig(level=loglevel)

        self._camera = Lepton(loglevel=loglevel)
        self._device_id = device
        if device is not None:
            self._camera.setup_video(device)

        self.logged_images: list = []
        self.logged_tstamps: list = []
        self.logged_cam_tstamps: list = []
        self.logged_frame_numbers: list = []
        self.enable_logging: bool = False
        self.timestamp_offset: float = 0.0

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_telemetry: Dict = {}
        self._lock = threading.Lock()
        self._running = threading.Event()

        self.start()

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        self.close()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background capture thread."""
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="LeptonCaptureThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        if not hasattr(self, "_running"):
            return
        self._running.clear()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logging.warning("LeptonWrapper: capture thread did not stop within timeout.")

    def close(self) -> None:
        """Stop the capture thread and release the camera."""
        self.stop()
        if hasattr(self, "_camera"):
            try:
                self._camera.release()
            except Exception as e:
                logging.warning(f"LeptonWrapper: error closing camera: {e}")

    # ------------------------------------------------------------------
    # Pause / resume for CCI access
    # ------------------------------------------------------------------

    def pause_capture(self) -> None:
        """Stop the capture thread and release the camera so CCI can use it."""
        self.stop()
        try:
            self._camera.release()
            self._camera.cap = None
        except Exception:
            pass

    def resume_capture(self) -> None:
        """Re-open the camera and restart the capture thread after CCI."""
        try:
            self._camera.setup_video(self._device_id)
        except Exception as e:
            logging.error(f"LeptonWrapper: failed to reopen camera: {e}")
            raise
        self.start()

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Continuously grab frames from the Lepton in a background thread.

        Grabs with telemetry rows intact, parses full telemetry, computes
        the timestamp offset on the first valid frame, then strips the
        telemetry rows so downstream consumers get clean thermal frames.
        Raw frames (with telemetry) are preserved for logging.
        """
        consecutive_failures = 0
        max_failures = 10
        offset_computed = False

        while self._running.is_set():
            try:
                raw_frame = self._camera.grab(strip_telemetry=False)
                if raw_frame is not None:
                    capture_time = time.time()

                    telemetry = self._parse_telemetry(raw_frame)
                    frame = raw_frame[:-2, :]

                    if not offset_computed:
                        cam_ts = telemetry.get("uptime_ms", 0) / 1000.0
                        if cam_ts > 0:
                            self.timestamp_offset = capture_time - cam_ts
                            offset_computed = True

                    with self._lock:
                        self._latest_frame = frame
                        self._latest_raw_frame = raw_frame
                        self._latest_telemetry = telemetry

                    self._post_capture_hook(
                        raw_frame, frame, capture_time, telemetry)
                    consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                logging.warning(
                    f"LeptonWrapper: frame grab failed ({consecutive_failures}/{max_failures}): {e}"
                )
                if consecutive_failures >= max_failures:
                    logging.error(
                        "LeptonWrapper: too many consecutive failures, stopping capture thread."
                    )
                    self._running.clear()
                    break

    # ------------------------------------------------------------------
    # Telemetry parsing (full row, IDD word layout)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_telemetry(raw_frame: np.ndarray) -> Dict:
        """
        Parse Telemetry Row A from a raw frame that still has its
        telemetry footer attached.  Uses the FLIR Lepton Software IDD
        (110-0144) word layout.
        """
        try:
            words = raw_frame[-2, :].view(np.uint16).copy()

            status = int(words[3]) << 16 | int(words[4])

            serial_upper = int(words[5]) << 16 | int(words[6])
            serial_lower = int(words[7]) << 16 | int(words[8])

            fpa_raw = int(words[19])
            housing_raw = int(words[20])

            ffc_elapsed = int(words[22]) << 16 | int(words[23])

            return {
                "telemetry_rev": int(words[0]),
                "uptime_ms": int(words[1]) << 16 | int(words[2]),
                "status": status,
                "ffc_desired": bool(status & (1 << 0)),
                "ffc_in_progress": bool(status & (1 << 1)),
                "agc_enabled": bool(status & (1 << 4)),
                "serial": (serial_upper << 32) | serial_lower,
                "frame_count": int(words[16]) << 16 | int(words[17]),
                "frame_mean": int(words[18]),
                "fpa_temp_c": (fpa_raw / 100.0 - 273.15) if fpa_raw else None,
                "housing_temp_c": (housing_raw / 100.0 - 273.15) if housing_raw else None,
                "ffc_elapsed_ms": ffc_elapsed,
                "agc_roi": (int(words[26]), int(words[27]),
                            int(words[28]), int(words[29]))
                           if len(words) > 29 else None,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Logging control
    # ------------------------------------------------------------------

    def _post_capture_hook(self, raw_frame: np.ndarray, frame: np.ndarray,
                           timestamp: float, telemetry: Dict) -> None:
        """
        Called from the capture thread after each successful grab.

        Logs the *raw* frame (with telemetry footer) so that recorded
        .npz files mirror the Boson format — telemetry rows are embedded
        in the frame data and can be parsed later during analysis.

        Args:
            raw_frame: Full frame including telemetry footer rows
            frame: Thermal-only image (telemetry stripped)
            timestamp: System timestamp (time.time()) at capture
            telemetry: Parsed telemetry dict for this frame
        """
        if self.enable_logging:
            self.logged_images.append(raw_frame)
            self.logged_tstamps.append(timestamp)
            cam_ts = telemetry.get("uptime_ms", 0) / 1000.0
            self.logged_cam_tstamps.append(cam_ts + self.timestamp_offset)
            self.logged_frame_numbers.append(telemetry.get("frame_count", 0))

    def start_logging(self) -> None:
        """Start accumulating frames and timestamps."""
        self.enable_logging = True

    def stop_logging(self) -> None:
        """Stop accumulating frames."""
        self.enable_logging = False

    def clear_logged_data(self) -> None:
        """Clear all previously logged data."""
        self.logged_images.clear()
        self.logged_tstamps.clear()
        self.logged_cam_tstamps.clear()
        self.logged_frame_numbers.clear()

    # ------------------------------------------------------------------
    # Frame / telemetry access
    # ------------------------------------------------------------------

    def get_next_image(self) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Return the most recently captured frame with telemetry, matching
        the BosonWithTelemetry.get_next_image() return signature.

        Returns:
            Tuple of:
            - image: Thermal image array (telemetry stripped)
            - timestamp: Camera-derived timestamp aligned to system clock
            - frame_number: Camera's internal frame counter
            - telemetry: Raw telemetry footer rows (2 × width, uint16)

        Raises:
            RuntimeError: If no frame has been captured yet.
        """
        with self._lock:
            frame = self._latest_frame
            raw_frame = self._latest_raw_frame
            telem = self._latest_telemetry.copy()
        if frame is None:
            raise RuntimeError(
                "No frame available yet. The camera may still be initializing."
            )

        cam_ts = telem.get("uptime_ms", 0) / 1000.0
        timestamp = cam_ts + self.timestamp_offset
        frame_number = telem.get("frame_count", 0)
        telemetry_rows = raw_frame[-2:, :] if raw_frame is not None else np.empty(0)

        return frame, timestamp, frame_number, telemetry_rows

    def get_telemetry(self) -> Dict:
        """Return a copy of the most recently parsed telemetry dictionary."""
        with self._lock:
            return self._latest_telemetry.copy()


# ======================================================================
# Live Viewer with HUD overlay  (run: python wrapper_lepton.py)
# ======================================================================

if __name__ == "__main__":
    import cv2
    import sys
    import os
    from datetime import datetime

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    COLORMAPS = [
        ("INFERNO",  cv2.COLORMAP_INFERNO),
        ("JET",      cv2.COLORMAP_JET),
        ("TURBO",    cv2.COLORMAP_TURBO),
        ("HOT",      cv2.COLORMAP_HOT),
        ("MAGMA",    cv2.COLORMAP_MAGMA),
        ("PLASMA",   cv2.COLORMAP_PLASMA),
        ("BONE",     cv2.COLORMAP_BONE),
        ("RAINBOW",  cv2.COLORMAP_RAINBOW),
        ("VIRIDIS",  cv2.COLORMAP_VIRIDIS),
    ]

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN
    COL_ON     = (0, 220, 100)    # green
    COL_OFF    = (60, 60, 200)    # red-ish
    COL_WARN   = (0, 180, 255)    # orange
    COL_WHITE  = (220, 220, 220)
    COL_CYAN   = (220, 200, 50)
    COL_DIM    = (140, 140, 140)
    COL_HEAD   = (255, 220, 100)
    HUD_BG     = (20, 20, 20)
    HUD_ALPHA  = 0.72
    MIN_CANVAS_W = 640

    # ------------------------------------------------------------------
    # CCI helper — try importing from lepton_filters
    # ------------------------------------------------------------------
    _cci_available = False
    _LeptonCCI = None
    _AGC_UNIT = _OEM_UNIT = _RAD_UNIT = _SYS_UNIT = _VID_UNIT = 0

    try:
        from lepton_filters import (
            LeptonCCI as _LeptonCCI,
            AGC_UNIT_ID as _AGC_UNIT,
            OEM_UNIT_ID as _OEM_UNIT,
            RAD_UNIT_ID as _RAD_UNIT,
            SYS_UNIT_ID as _SYS_UNIT,
            VID_UNIT_ID as _VID_UNIT,
            _load_libuvc,
        )
        if _load_libuvc() is not None:
            _cci_available = True
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Camera-filter key bindings
    #   key: (shortcut_label, unit_id, control_id, kind, extra)
    #   kind = "toggle" | "cycle" | "run"
    # ------------------------------------------------------------------

    CAMERA_KEYS: Dict[int, tuple] = {}

    if _cci_available:
        CAMERA_KEYS = {
            ord("1"): ("AGC",          _AGC_UNIT, 1,  "toggle", None),
            ord("2"): ("AGC Policy",   _AGC_UNIT, 2,  "cycle",  [0, 1]),
            ord("3"): ("Radiometry",   _RAD_UNIT, 5,  "toggle", None),
            ord("4"): ("TLinear",      _RAD_UNIT, 37, "toggle", None),
            ord("5"): ("Gain Mode",    _SYS_UNIT, 17, "cycle",  [0, 1, 2]),
            ord("6"): ("SBNUC",        _VID_UNIT, 8,  "toggle", None),
            ord("7"): ("Bad Pixel",    _OEM_UNIT, 27, "toggle", None),
            ord("8"): ("Temporal NF",  _OEM_UNIT, 28, "toggle", None),
            ord("9"): ("Col Noise",    _OEM_UNIT, 29, "toggle", None),
            ord("0"): ("Pixel Noise",  _OEM_UNIT, 30, "toggle", None),
            ord("g"): ("Gamma",        _OEM_UNIT, 13, "toggle", None),
            ord("p"): ("Polarity",     _VID_UNIT, 1,  "cycle",  [0, 1]),
            ord("f"): ("Run FFC",      _RAD_UNIT, 12, "run",    None),
        }

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def put(img, text, x, y, color=COL_WHITE, scale=0.42, thick=1, font=FONT):
        cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    def draw_rect_alpha(img, pt1, pt2, color, alpha):
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # ------------------------------------------------------------------
    # HUD renderer
    # ------------------------------------------------------------------

    def draw_hud(canvas, telem, cmap_name, cmap_idx, show_hud, fps,
                 recording, cci_ok, cci_msg, frame_shape):
        h, w = canvas.shape[:2]
        line_h = 18
        margin = 8

        # ── Top bar (always visible) ────────────────────────────────
        draw_rect_alpha(canvas, (0, 0), (w, 28), HUD_BG, HUD_ALPHA)

        res_str = f"{frame_shape[1]}x{frame_shape[0]}" if frame_shape else "?"
        top_left = f"FLIR Lepton  {res_str}"
        put(canvas, top_left, margin, 18, COL_HEAD, 0.45, 1)

        fps_str = f"{fps:.1f} FPS"
        put(canvas, fps_str, w - 90, 18, COL_CYAN, 0.42)

        if recording:
            cv2.circle(canvas, (w - 108, 14), 5, (0, 0, 255), -1)

        # ── Bottom bar (always visible) ─────────────────────────────
        bot_y = h - 6
        draw_rect_alpha(canvas, (0, h - 26), (w, h), HUD_BG, HUD_ALPHA)

        fpa = telem.get("fpa_temp_c")
        fpa_s = f"FPA:{fpa:.1f}C" if fpa is not None else "FPA:--"
        housing = telem.get("housing_temp_c")
        hsg_s = f"Hsg:{housing:.1f}C" if housing is not None else "Hsg:--"
        ffc_s = "FFC:RUN" if telem.get("ffc_in_progress") else "FFC:ok"
        ffc_col = COL_WARN if telem.get("ffc_in_progress") else COL_DIM
        put(canvas, fpa_s, margin, bot_y, COL_WHITE, 0.40)
        put(canvas, hsg_s, 110, bot_y, COL_WHITE, 0.40)
        put(canvas, ffc_s, 220, bot_y, ffc_col, 0.40)

        hint = "[H]ud [C]map [Q]uit"
        put(canvas, hint, w - 188, bot_y, COL_DIM, 0.36)

        if not show_hud:
            return

        # ── Side panel ──────────────────────────────────────────────
        pw = 236
        px = w - pw - 2
        py = 32
        panel_h = h - 62
        draw_rect_alpha(canvas, (px, py), (px + pw, py + panel_h), HUD_BG, HUD_ALPHA)

        cx = px + margin
        cy = py + line_h

        # -- Camera filters section --
        put(canvas, "CAMERA FILTERS", cx, cy, COL_HEAD, 0.40, 1)
        cy += 4
        cv2.line(canvas, (cx, cy), (cx + pw - 2 * margin, cy), (80, 80, 80), 1)
        cy += line_h

        agc = telem.get("agc_enabled")
        agc_s = "ON" if agc else "OFF" if agc is not None else "?"
        agc_c = COL_ON if agc else COL_OFF

        filter_lines = [
            ("[1] AGC",       agc_s, agc_c),
            ("[2] AGC Policy", "--",  COL_DIM),
            ("[3] Radiometry", "--",  COL_DIM),
            ("[4] TLinear",    "--",  COL_DIM),
            ("[5] Gain Mode",  "--",  COL_DIM),
            ("[6] SBNUC",      "--",  COL_DIM),
            ("[7] Bad Pixel",  "--",  COL_DIM),
            ("[8] Temporal NF", "--", COL_DIM),
            ("[9] Col Noise",  "--",  COL_DIM),
            ("[0] Pixel Noise", "--", COL_DIM),
            ("[G] Gamma",      "--",  COL_DIM),
            ("[P] Polarity",   "--",  COL_DIM),
        ]

        for label, val, col in filter_lines:
            put(canvas, label, cx, cy, COL_WHITE, 0.36)
            put(canvas, val, cx + 130, cy, col, 0.36, 1)
            cy += line_h

        # FFC line
        cy += 2
        ffc_state = "RUNNING" if telem.get("ffc_in_progress") else "Idle"
        ffc_elapsed = telem.get("ffc_elapsed_ms", 0)
        ffc_str = f"{ffc_state} ({ffc_elapsed / 1000:.0f}s ago)"
        put(canvas, "[F] FFC", cx, cy, COL_WHITE, 0.36)
        ffc_col_v = COL_WARN if telem.get("ffc_in_progress") else COL_ON
        put(canvas, ffc_str, cx + 130, cy, ffc_col_v, 0.36)
        cy += line_h + 6

        # -- Display section --
        cv2.line(canvas, (cx, cy - 4), (cx + pw - 2 * margin, cy - 4), (80, 80, 80), 1)
        put(canvas, "DISPLAY", cx, cy + 2, COL_HEAD, 0.40, 1)
        cy += line_h + 2

        put(canvas, "[C] Colormap", cx, cy, COL_WHITE, 0.36)
        put(canvas, cmap_name, cx + 130, cy, COL_CYAN, 0.36, 1)
        cy += line_h

        put(canvas, "[R] Record", cx, cy, COL_WHITE, 0.36)
        rec_str = "REC" if recording else "OFF"
        rec_col = (0, 0, 255) if recording else COL_DIM
        put(canvas, rec_str, cx + 130, cy, rec_col, 0.36, 1)
        cy += line_h

        put(canvas, "[Space] Screenshot", cx, cy, COL_DIM, 0.36)
        cy += line_h + 6

        # -- CCI status --
        cv2.line(canvas, (cx, cy - 4), (cx + pw - 2 * margin, cy - 4), (80, 80, 80), 1)
        cci_label = "CCI: Ready" if cci_ok else "CCI: Unavailable"
        cci_col = COL_ON if cci_ok else COL_OFF
        put(canvas, cci_label, cx, cy + 2, cci_col, 0.36)
        cy += line_h

        if cci_msg:
            for msg_line in cci_msg.split("\n"):
                put(canvas, msg_line, cx, cy, COL_WARN, 0.33)
                cy += line_h - 2

    # ------------------------------------------------------------------
    # HUD with live CCI state overlay (updates from CCI reads)
    # ------------------------------------------------------------------

    def update_hud_from_cci(canvas, cci_states, px, py, margin, line_h):
        """Overwrite the '--' placeholders with real CCI-read values."""
        cx = px + margin

        val_x = cx + 130
        cy_start = py + 18 + 4 + 18  # after header + line + first entry

        labels_map = {
            0:  ("agc_enable",   {0: "OFF", 1: "ON"}),
            1:  ("agc_policy",   {0: "LIN", 1: "HEQ"}),
            2:  ("radiometry",   {0: "OFF", 1: "ON"}),
            3:  ("tlinear",      {0: "OFF", 1: "ON"}),
            4:  ("gain_mode",    {0: "HIGH", 1: "LOW", 2: "AUTO"}),
            5:  ("sbnuc",        {0: "OFF", 1: "ON"}),
            6:  ("bad_pixel",    {0: "OFF", 1: "ON"}),
            7:  ("temporal_nf",  {0: "OFF", 1: "ON"}),
            8:  ("col_noise",    {0: "OFF", 1: "ON"}),
            9:  ("pixel_noise",  {0: "OFF", 1: "ON"}),
            10: ("gamma",        {0: "OFF", 1: "ON"}),
            11: ("polarity",     {0: "WHOT", 1: "BHOT"}),
        }

        for idx, (key, val_map) in labels_map.items():
            cy = cy_start + idx * line_h
            raw = cci_states.get(key)
            if raw is not None:
                text = val_map.get(raw, str(raw))
                col = COL_ON if raw else COL_OFF
                if key == "gain_mode":
                    col = COL_CYAN
                elif key in ("agc_policy", "polarity"):
                    col = COL_CYAN if raw else COL_WHITE
                draw_rect_alpha(canvas, (val_x - 2, cy - 12), (val_x + 80, cy + 4), HUD_BG, 0.90)
                put(canvas, text, val_x, cy, col, 0.36, 1)

    # ------------------------------------------------------------------
    # CCI read/write helpers
    # ------------------------------------------------------------------

    def cci_read_all(wrapper):
        """Pause video, open CCI, read all key filter states, resume."""
        states = {}
        try:
            wrapper.pause_capture()
            time.sleep(0.15)
            with _LeptonCCI() as cci:
                read_list = [
                    ("agc_enable",  _AGC_UNIT, 1),
                    ("agc_policy",  _AGC_UNIT, 2),
                    ("radiometry",  _RAD_UNIT, 5),
                    ("tlinear",     _RAD_UNIT, 37),
                    ("gain_mode",   _SYS_UNIT, 17),
                    ("sbnuc",       _VID_UNIT, 8),
                    ("bad_pixel",   _OEM_UNIT, 27),
                    ("temporal_nf", _OEM_UNIT, 28),
                    ("col_noise",   _OEM_UNIT, 29),
                    ("pixel_noise", _OEM_UNIT, 30),
                    ("gamma",       _OEM_UNIT, 13),
                    ("polarity",    _VID_UNIT, 1),
                ]
                for key, unit, cid in read_list:
                    try:
                        states[key] = cci.get_u32(unit, cid)
                    except Exception:
                        pass
        except Exception as e:
            logging.warning(f"CCI read failed: {e}")
        finally:
            try:
                wrapper.resume_capture()
            except Exception as e:
                logging.error(f"Failed to resume capture after CCI: {e}")
        return states

    def cci_toggle(wrapper, unit, cid, kind, extra):
        """Pause video, toggle/cycle/run a CCI control, resume. Returns new value or None."""
        new_val = None
        try:
            wrapper.pause_capture()
            time.sleep(0.15)
            with _LeptonCCI() as cci:
                if kind == "run":
                    from lepton_filters import _cci_run
                    _cci_run(cci.devh, unit, cid)
                    return 0
                cur = cci.get_u32(unit, cid)
                if kind == "toggle":
                    new_val = 0 if cur else 1
                elif kind == "cycle" and extra:
                    idx = extra.index(cur) if cur in extra else -1
                    new_val = extra[(idx + 1) % len(extra)]
                if new_val is not None:
                    cci.set_u32(unit, cid, new_val)
        except Exception as e:
            logging.warning(f"CCI toggle failed: {e}")
            new_val = None
        finally:
            try:
                wrapper.resume_capture()
            except Exception as e:
                logging.error(f"Failed to resume after CCI toggle: {e}")
        return new_val

    # ------------------------------------------------------------------
    # Main viewer loop
    # ------------------------------------------------------------------

    def run_live_viewer():
        print("=" * 52)
        print("  FLIR Lepton Live Viewer")
        print("=" * 52)
        print("  Press [H] to toggle HUD overlay")
        print("  Press [Q] to quit")
        if _cci_available:
            print("  CCI (libuvc) detected — filter toggles enabled")
        else:
            print("  CCI not available — display + telemetry only")
            print("  (Use lepton_filters.py --libuvc for filter control)")
        print("=" * 52)

        lepton: Optional[LeptonWrapper] = None
        try:
            lepton = LeptonWrapper()
            cv2.namedWindow("Lepton Live", cv2.WINDOW_NORMAL)

            # Wait for first frame
            for _ in range(60):
                with lepton._lock:
                    ready = lepton._latest_frame is not None
                if ready:
                    break
                time.sleep(0.1)

            # Display state
            cmap_idx = 0
            show_hud = True
            recording = False
            cci_msg = ""
            cci_msg_until = 0.0
            cci_states: Dict = {}
            frame_shape = (0, 0)

            # FPS tracking
            fps = 0.0
            fps_counter = 0
            fps_timer = time.time()

            # Try initial CCI read
            if _cci_available:
                cci_msg = "Reading filters..."
                cci_states = cci_read_all(lepton)
                if cci_states:
                    cci_msg = ""
                else:
                    cci_msg = "CCI read failed.\nFilters shown from\ntelemetry only."
                    cci_msg_until = time.time() + 5.0

            while True:
                try:
                    image, timestamp, frame_number, _ = lepton.get_next_image()
                except RuntimeError:
                    time.sleep(0.05)
                    continue

                telem = lepton.get_telemetry()
                frame_shape = image.shape

                # Scale up for readable HUD text
                ih, iw = image.shape[:2]
                scale = max(1, MIN_CANVAS_W // iw)
                canvas_w = iw * scale
                canvas_h = ih * scale

                img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cmap_name, cmap_val = COLORMAPS[cmap_idx]
                colorized = cv2.applyColorMap(img_8bit, cmap_val)
                canvas = cv2.resize(colorized, (canvas_w, canvas_h),
                                    interpolation=cv2.INTER_NEAREST)

                # Min / max pixel temperature hints
                raw_min = float(np.min(image))
                raw_max = float(np.max(image))
                min_loc = np.unravel_index(np.argmin(image), image.shape)
                max_loc = np.unravel_index(np.argmax(image), image.shape)
                cv2.drawMarker(canvas,
                               (int(min_loc[1] * scale), int(min_loc[0] * scale)),
                               (255, 100, 0), cv2.MARKER_CROSS, 10, 1)
                cv2.drawMarker(canvas,
                               (int(max_loc[1] * scale), int(max_loc[0] * scale)),
                               (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                put(canvas, f"{raw_min:.0f}",
                    int(min_loc[1] * scale) + 6, int(min_loc[0] * scale) - 4,
                    (255, 100, 0), 0.34)
                put(canvas, f"{raw_max:.0f}",
                    int(max_loc[1] * scale) + 6, int(max_loc[0] * scale) - 4,
                    (0, 0, 255), 0.34)

                # FPS
                fps_counter += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # Clear expired CCI message
                if cci_msg and cci_msg_until and now > cci_msg_until:
                    cci_msg = ""

                # Draw HUD
                cci_ok = _cci_available and bool(cci_states)
                draw_hud(canvas, telem, cmap_name, cmap_idx, show_hud,
                         fps, recording, cci_ok, cci_msg, frame_shape)

                # Overlay CCI-read filter values when available
                if show_hud and cci_states:
                    pw = 236
                    px = canvas_w - pw - 2
                    py = 32
                    update_hud_from_cci(canvas, cci_states, px, py, 8, 18)

                cv2.imshow("Lepton Live", canvas)

                # --- Keyboard handling ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                elif key == ord("h"):
                    show_hud = not show_hud

                elif key == ord("c"):
                    cmap_idx = (cmap_idx + 1) % len(COLORMAPS)

                elif key == ord("r"):
                    recording = not recording
                    if recording:
                        lepton.clear_logged_data()
                        lepton.start_logging()
                        cci_msg = "Recording started"
                        cci_msg_until = time.time() + 2.0
                    else:
                        lepton.stop_logging()
                        n = len(lepton.logged_images)
                        os.makedirs("lepton_recordings", exist_ok=True)
                        fname = f"lepton_recordings/live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                        if n > 0:
                            np.savez(
                                fname,
                                raw_thr_frames=np.array(lepton.logged_images),
                                raw_thr_tstamps=np.array(lepton.logged_tstamps),
                                thr_cam_timestamp_offset=lepton.timestamp_offset,
                                raw_thr_cam_tstamps=np.array(lepton.logged_cam_tstamps),
                                raw_thr_frame_numbers=np.array(lepton.logged_frame_numbers),
                            )
                            cci_msg = f"Saved {n} frames: {fname}"
                        else:
                            cci_msg = "No frames recorded"
                        cci_msg_until = time.time() + 4.0

                elif key == ord(" "):
                    os.makedirs("lepton_recordings", exist_ok=True)
                    fname = f"lepton_recordings/snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(fname, canvas)
                    cci_msg = f"Screenshot: {fname}"
                    cci_msg_until = time.time() + 3.0

                elif key in CAMERA_KEYS and _cci_available:
                    label, unit, cid, kind, extra = CAMERA_KEYS[key]
                    cci_msg = f"CCI: {label}..."
                    cci_msg_until = 0

                    result = cci_toggle(lepton, unit, cid, kind, extra)
                    if result is not None:
                        cci_states = cci_read_all(lepton)
                        cci_msg = f"{label}: OK"
                        cci_msg_until = time.time() + 2.0
                    else:
                        cci_msg = f"{label}: FAILED"
                        cci_msg_until = time.time() + 3.0

                elif key in CAMERA_KEYS and not _cci_available:
                    cci_msg = "CCI unavailable.\nUse lepton_filters.py"
                    cci_msg_until = time.time() + 3.0

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure the Lepton camera is connected and accessible.")

        finally:
            if lepton is not None:
                if recording:
                    lepton.stop_logging()
                lepton.close()
            cv2.destroyAllWindows()
            print("Camera stopped. Windows closed.")

    run_live_viewer()
