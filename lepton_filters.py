"""
FLIR Lepton On-Board Filter Control

Query and control the on-board image processing filters inside the FLIR
Lepton camera module via the PureThermal board's UVC extension units.

The Lepton has 5 CCI (Command and Control Interface) modules that expose
controllable filters and processing stages:

  Module   │ Unit │ Filters / Controls
  ─────────┼──────┼──────────────────────────────────────────────────
  AGC      │  3   │ Automatic Gain Control (HEQ / Linear modes)
  OEM      │  4   │ Bad-pixel replace, temporal / column / pixel
           │      │ noise filters, gamma, video output
  RAD      │  5   │ Radiometry, TLinear, spotmeter, noise-filter
           │      │ scale factors (CNF / TNF / SNF)
  SYS      │  6   │ Telemetry, FFC, gain mode (High/Low/Auto)
  VID      │  7   │ Polarity, LUT / palette, SBNUC, focus calc,
           │      │ gamma, freeze, boresight

Two operating modes:
  1. Telemetry mode (default) — reads filter states from the frame
     telemetry rows using flirpy.  Works immediately on all platforms
     without any extra setup.
  2. libuvc mode (--libuvc) — full get/set CCI access through UVC
     extension units.  Requires the libuvc shared library.

Usage examples:
    python lepton_filters.py                    # list filters via telemetry
    python lepton_filters.py --libuvc           # list filters via CCI
    python lepton_filters.py --libuvc --set agc on
    python lepton_filters.py --libuvc --set agc off
    python lepton_filters.py --libuvc --set agc_policy heq
    python lepton_filters.py --libuvc --set tlinear on
    python lepton_filters.py --libuvc --set gain_mode auto
    python lepton_filters.py --libuvc --run ffc
"""

import argparse
import sys
import time
import struct
import logging
from ctypes import (
    cdll, c_void_p, c_uint8, c_uint16, c_uint32, c_int, c_ulong,
    c_size_t, c_long, c_ubyte, c_ushort, c_char, c_int32,
    Structure, POINTER, CFUNCTYPE, byref, create_string_buffer, cast,
    pointer, sizeof,
)
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# UVC extension-unit IDs (PureThermal firmware)
# ---------------------------------------------------------------------------
AGC_UNIT_ID = 3
OEM_UNIT_ID = 4
RAD_UNIT_ID = 5
SYS_UNIT_ID = 6
VID_UNIT_ID = 7

PT_USB_VID = 0x1E4E
PT_USB_PID = 0x0100

# ---------------------------------------------------------------------------
# Complete filter / control catalogue
# ---------------------------------------------------------------------------
FILTER_CATALOGUE = {
    "AGC": {
        "unit": AGC_UNIT_ID,
        "controls": {
            "AGC Enable State":             {"cid": 1,  "rw": "rw", "size": 4,  "desc": "Master AGC on/off (0=off, 1=on)"},
            "AGC Policy":                   {"cid": 2,  "rw": "rw", "size": 4,  "desc": "0=Linear, 1=HEQ (Histogram Equalization)"},
            "AGC ROI":                      {"cid": 3,  "rw": "rw", "size": 8,  "desc": "Region of interest (top, left, bottom, right)"},
            "AGC Statistics":               {"cid": 4,  "rw": "r",  "size": 8,  "desc": "Min/max/mean/pixel-count in ROI"},
            "AGC Histogram Clip Percent":   {"cid": 5,  "rw": "rw", "size": 4,  "desc": "Histogram clipping percentage"},
            "AGC Histogram Tail Size":      {"cid": 6,  "rw": "rw", "size": 4,  "desc": "Histogram tail size"},
            "AGC Linear Max Gain":          {"cid": 7,  "rw": "rw", "size": 4,  "desc": "Max gain in linear AGC mode"},
            "AGC Linear Midpoint":          {"cid": 8,  "rw": "rw", "size": 4,  "desc": "Linear AGC midpoint"},
            "AGC Linear Dampening Factor":  {"cid": 9,  "rw": "rw", "size": 4,  "desc": "Linear AGC dampening"},
            "AGC HEQ Dampening Factor":     {"cid": 10, "rw": "rw", "size": 4,  "desc": "HEQ dampening (0–256)"},
            "AGC HEQ Max Gain":             {"cid": 11, "rw": "rw", "size": 4,  "desc": "Max gain in HEQ mode"},
            "AGC HEQ Clip Limit High":      {"cid": 12, "rw": "rw", "size": 4,  "desc": "HEQ upper clip limit (4800 default)"},
            "AGC HEQ Clip Limit Low":       {"cid": 13, "rw": "rw", "size": 4,  "desc": "HEQ lower clip limit (512 default)"},
            "AGC HEQ Bin Extension":        {"cid": 14, "rw": "rw", "size": 4,  "desc": "HEQ bin extension"},
            "AGC HEQ Midpoint":             {"cid": 15, "rw": "rw", "size": 4,  "desc": "HEQ midpoint"},
            "AGC HEQ Empty Counts":         {"cid": 16, "rw": "rw", "size": 4,  "desc": "HEQ empty histogram bin count"},
            "AGC HEQ Normalization Factor": {"cid": 17, "rw": "rw", "size": 4,  "desc": "HEQ normalization factor"},
            "AGC HEQ Scale Factor":         {"cid": 18, "rw": "rw", "size": 4,  "desc": "HEQ output scale (0=8-bit, 1=14-bit)"},
            "AGC Calc Enable State":        {"cid": 19, "rw": "rw", "size": 4,  "desc": "Continuous AGC calculation on/off"},
            "AGC HEQ Linear Percent":       {"cid": 20, "rw": "rw", "size": 4,  "desc": "HEQ linear-to-HEQ blend percent"},
        },
    },

    "OEM": {
        "unit": OEM_UNIT_ID,
        "controls": {
            "Video Output Enable":            {"cid": 9,  "rw": "rw", "size": 4,  "desc": "Enable/disable video output"},
            "Video Output Format":            {"cid": 10, "rw": "rw", "size": 4,  "desc": "Output format (RAW14, RGB888, ...)"},
            "Video Output Source":            {"cid": 11, "rw": "rw", "size": 4,  "desc": "0=RAW, 1=Cooked, 2=Ramp, 3=Constant"},
            "Video Gamma Enable":             {"cid": 13, "rw": "rw", "size": 4,  "desc": "Gamma correction on/off"},
            "Bad Pixel Replace Control":      {"cid": 27, "rw": "rw", "size": 4,  "desc": "Bad-pixel replacement filter on/off"},
            "Temporal Filter Control":        {"cid": 28, "rw": "rw", "size": 4,  "desc": "OEM temporal noise filter on/off"},
            "Column Noise Estimate Control":  {"cid": 29, "rw": "rw", "size": 4,  "desc": "Column noise estimate filter on/off"},
            "Pixel Noise Estimate Control":   {"cid": 30, "rw": "rw", "size": 4,  "desc": "Pixel noise estimate filter on/off"},
        },
    },

    "RAD": {
        "unit": RAD_UNIT_ID,
        "controls": {
            "Radiometry Enable State":    {"cid": 5,  "rw": "rw", "size": 4,  "desc": "Master radiometry on/off"},
            "TLinear Enable State":       {"cid": 37, "rw": "rw", "size": 4,  "desc": "TLinear output mode on/off"},
            "TLinear Resolution":         {"cid": 38, "rw": "rw", "size": 4,  "desc": "TLinear resolution (0=0.1K, 1=0.01K)"},
            "TLinear Auto Resolution":    {"cid": 39, "rw": "rw", "size": 4,  "desc": "Auto-select TLinear resolution"},
            "Spotmeter ROI":              {"cid": 40, "rw": "rw", "size": 8,  "desc": "Spotmeter region of interest"},
            "Spotmeter Obj Kelvin":       {"cid": 41, "rw": "r",  "size": 4,  "desc": "Spotmeter object temperature (K×100)"},
            "Radiometry Filter":          {"cid": 23, "rw": "rw", "size": 4,  "desc": "Radiometric temporal filter setting"},
            "CNF Scale Factor":           {"cid": 34, "rw": "rw", "size": 4,  "desc": "Column Noise Filter scale factor"},
            "TNF Scale Factor":           {"cid": 35, "rw": "rw", "size": 4,  "desc": "Temporal Noise Filter scale factor"},
            "SNF Scale Factor":           {"cid": 36, "rw": "rw", "size": 4,  "desc": "Spatial Noise Filter scale factor"},
            "Run FFC":                    {"cid": 12, "rw": "run","size": 0,  "desc": "Trigger flat-field correction now"},
        },
    },

    "SYS": {
        "unit": SYS_UNIT_ID,
        "controls": {
            "Telemetry Enable State":  {"cid": 7,  "rw": "rw", "size": 4,  "desc": "Embed telemetry rows on/off"},
            "Telemetry Location":      {"cid": 8,  "rw": "rw", "size": 4,  "desc": "0=Header (top), 1=Footer (bottom)"},
            "FFC Shutter Mode":        {"cid": 14, "rw": "rw", "size": 20, "desc": "FFC mode object (manual/auto/external)"},
            "Gain Mode":               {"cid": 17, "rw": "rw", "size": 4,  "desc": "0=High, 1=Low, 2=Auto"},
            "Shutter Position":        {"cid": 13, "rw": "rw", "size": 4,  "desc": "0=Unknown, 1=Idle, 2=Open, 3=Closed, 4=BrakeOn"},
        },
    },

    "VID": {
        "unit": VID_UNIT_ID,
        "controls": {
            "Polarity Select":          {"cid": 1,  "rw": "rw", "size": 4,  "desc": "0=WhiteHot, 1=BlackHot"},
            "LUT Select":               {"cid": 2,  "rw": "rw", "size": 4,  "desc": "Pseudo-color palette (0=Wheel6, 1=Fusion, ...)"},
            "Focus Calc Enable":        {"cid": 4,  "rw": "rw", "size": 4,  "desc": "Focus metric calculation on/off"},
            "Focus ROI":                {"cid": 5,  "rw": "rw", "size": 8,  "desc": "Focus calculation region"},
            "Focus Threshold":          {"cid": 6,  "rw": "rw", "size": 4,  "desc": "Focus metric threshold"},
            "Focus Metric":             {"cid": 7,  "rw": "r",  "size": 4,  "desc": "Current focus metric value"},
            "SBNUC Enable":             {"cid": 8,  "rw": "rw", "size": 4,  "desc": "Scene-Based NUC correction on/off"},
            "Gamma Select":             {"cid": 9,  "rw": "rw", "size": 4,  "desc": "Gamma correction table select"},
            "Freeze Enable":            {"cid": 10, "rw": "rw", "size": 4,  "desc": "Freeze current frame output on/off"},
            "Boresight Calc Enable":    {"cid": 11, "rw": "rw", "size": 4,  "desc": "Boresight calculation on/off"},
            "Video Output Format":      {"cid": 13, "rw": "rw", "size": 4,  "desc": "Video output format select"},
        },
    },
}


# ---------------------------------------------------------------------------
# libuvc ctypes wrappers (loaded only when --libuvc is used)
# ---------------------------------------------------------------------------
_libuvc = None


class _uvc_context(Structure):
    _fields_ = [("usb_ctx", c_void_p), ("own_usb_ctx", c_uint8),
                ("open_devices", c_void_p), ("handler_thread", c_ulong),
                ("kill_handler_thread", c_int)]


class _uvc_device(Structure):
    _fields_ = [("ctx", POINTER(_uvc_context)), ("ref", c_int),
                ("usb_dev", c_void_p)]


class _uvc_device_handle(Structure):
    _fields_ = [("dev", POINTER(_uvc_device)),
                ("prev", c_void_p), ("next", c_void_p),
                ("usb_devh", c_void_p), ("info", c_void_p),
                ("status_xfer", c_void_p), ("status_buf", c_ubyte * 32),
                ("status_cb", c_void_p), ("status_user_ptr", c_void_p),
                ("button_cb", c_void_p), ("button_user_ptr", c_void_p),
                ("streams", c_void_p), ("is_isight", c_ubyte)]


def _load_libuvc():
    """Load the libuvc shared library, return the cdll handle or None."""
    global _libuvc
    if _libuvc is not None:
        return _libuvc

    import platform
    names = {
        "Linux":   "libuvc.so",
        "Darwin":  "libuvc.dylib",
        "Windows": "libuvc",
    }
    lib_name = names.get(platform.system(), "libuvc")
    try:
        _libuvc = cdll.LoadLibrary(lib_name)
        return _libuvc
    except OSError:
        return None


def _cci_get(devh, unit_id: int, control_id: int, size: int) -> bytes:
    """Read *size* bytes from a UVC extension-unit control (GET_CUR)."""
    buf = create_string_buffer(size)
    ret = _libuvc.uvc_get_ctrl(devh, unit_id, control_id, byref(buf), size, 0x81)
    if ret < 0:
        raise RuntimeError(f"uvc_get_ctrl failed (unit={unit_id}, cid={control_id}, ret={ret})")
    return buf.raw[:size]


def _cci_set(devh, unit_id: int, control_id: int, data: bytes):
    """Write bytes to a UVC extension-unit control (SET_CUR)."""
    buf = create_string_buffer(data, len(data))
    ret = _libuvc.uvc_set_ctrl(devh, unit_id, control_id, byref(buf), len(data))
    if ret < 0:
        raise RuntimeError(f"uvc_set_ctrl failed (unit={unit_id}, cid={control_id}, ret={ret})")


def _cci_run(devh, unit_id: int, control_id: int):
    """Execute a RUN command (SET with empty/dummy payload)."""
    buf = create_string_buffer(2)
    _libuvc.uvc_set_ctrl(devh, unit_id, control_id, byref(buf), 2)


# ---------------------------------------------------------------------------
# High-level helpers for the most useful filter toggles
# ---------------------------------------------------------------------------
SHORTCUT_MAP = {
    "agc":                ("AGC",  "AGC Enable State"),
    "agc_policy":         ("AGC",  "AGC Policy"),
    "agc_calc":           ("AGC",  "AGC Calc Enable State"),
    "tlinear":            ("RAD",  "TLinear Enable State"),
    "radiometry":         ("RAD",  "Radiometry Enable State"),
    "rad_filter":         ("RAD",  "Radiometry Filter"),
    "cnf":                ("RAD",  "CNF Scale Factor"),
    "tnf":                ("RAD",  "TNF Scale Factor"),
    "snf":                ("RAD",  "SNF Scale Factor"),
    "bad_pixel":          ("OEM",  "Bad Pixel Replace Control"),
    "temporal_filter":    ("OEM",  "Temporal Filter Control"),
    "column_noise":       ("OEM",  "Column Noise Estimate Control"),
    "pixel_noise":        ("OEM",  "Pixel Noise Estimate Control"),
    "gamma":              ("OEM",  "Video Gamma Enable"),
    "sbnuc":              ("VID",  "SBNUC Enable"),
    "freeze":             ("VID",  "Freeze Enable"),
    "focus_calc":         ("VID",  "Focus Calc Enable"),
    "boresight":          ("VID",  "Boresight Calc Enable"),
    "telemetry":          ("SYS",  "Telemetry Enable State"),
    "gain_mode":          ("SYS",  "Gain Mode"),
    "polarity":           ("VID",  "Polarity Select"),
    "lut":                ("VID",  "LUT Select"),
}

VALUE_ALIASES = {
    "on": 1, "off": 0, "enable": 1, "disable": 0,
    "true": 1, "false": 0, "1": 1, "0": 0,
    "heq": 1, "linear": 0,
    "high": 0, "low": 1, "auto": 2,
    "whitehot": 0, "blackhot": 1,
    "header": 0, "footer": 1,
}


# ---------------------------------------------------------------------------
# Telemetry-based filter state reading (works without libuvc)
# ---------------------------------------------------------------------------
def read_filter_states_from_telemetry(device: Optional[int] = None) -> dict:
    """Grab a frame via flirpy and extract filter states from the telemetry row."""
    from flirpy.camera.lepton import Lepton as FLIRLepton

    cam = FLIRLepton()
    cam.setup_video(device)

    frame = None
    for _ in range(30):
        f = cam.grab(strip_telemetry=False)
        if f is not None and f.size > 0:
            frame = f
            break
        time.sleep(0.15)

    cam.release()

    if frame is None:
        raise RuntimeError("Could not capture a frame from the Lepton.")

    words = frame[-2, :].view(np.uint16).copy()
    h, w = frame.shape[:2]

    status = int(words[3]) << 16 | int(words[4])

    states = {}
    states["resolution"] = f"{w}×{h - 2}"
    states["AGC Enabled"] = bool(status & (1 << 4))
    states["FFC State"] = "In Progress" if (status & (1 << 1)) else "Idle"
    states["FFC Desired"] = bool(status & (1 << 0))

    fpa_raw = int(words[19])
    if fpa_raw:
        states["FPA Temperature"] = f"{fpa_raw / 100.0 - 273.15:.2f} °C"

    housing_raw = int(words[20])
    if housing_raw:
        states["Housing Temperature"] = f"{housing_raw / 100.0 - 273.15:.2f} °C"

    states["Frame Counter"] = int(words[16]) << 16 | int(words[17])
    states["Frame Mean"] = int(words[18])

    if len(words) > 33:
        states["AGC ROI (T,L,B,R)"] = (int(words[26]), int(words[27]),
                                        int(words[28]), int(words[29]))
        states["Video Output Format (raw)"] = int(words[32]) << 16 | int(words[33])

    ffc_elapsed = int(words[22]) << 16 | int(words[23])
    states["Time Since Last FFC"] = f"{ffc_elapsed} ms ({ffc_elapsed / 1000:.1f} s)"

    return states


# ---------------------------------------------------------------------------
# libuvc-based CCI control session
# ---------------------------------------------------------------------------
class LeptonCCI:
    """Context manager that opens the PureThermal device via libuvc for CCI."""

    def __init__(self):
        lib = _load_libuvc()
        if lib is None:
            raise RuntimeError(
                "libuvc not found.  Install it for full CCI access.\n"
                "  Linux  : sudo apt install libuvc-dev\n"
                "  macOS  : brew install libuvc\n"
                "  Windows: build from https://github.com/groupgets/libuvc\n"
                "           or use --telemetry mode instead."
            )
        self._ctx = POINTER(_uvc_context)()
        self._dev = POINTER(_uvc_device)()
        self._devh = POINTER(_uvc_device_handle)()

    def __enter__(self):
        lib = _libuvc
        rc = lib.uvc_init(byref(self._ctx), 0)
        if rc < 0:
            raise RuntimeError(f"uvc_init failed ({rc})")

        rc = lib.uvc_find_device(self._ctx, byref(self._dev),
                                 PT_USB_VID, PT_USB_PID, 0)
        if rc < 0:
            lib.uvc_exit(self._ctx)
            raise RuntimeError(
                f"PureThermal device not found (VID=0x{PT_USB_VID:04X}, "
                f"PID=0x{PT_USB_PID:04X}).  Is the camera plugged in?")

        rc = lib.uvc_open(self._dev, byref(self._devh))
        if rc < 0:
            lib.uvc_unref_device(self._dev)
            lib.uvc_exit(self._ctx)
            raise RuntimeError(f"uvc_open failed ({rc})")

        return self

    def __exit__(self, *exc):
        _libuvc.uvc_close(self._devh)
        _libuvc.uvc_unref_device(self._dev)
        _libuvc.uvc_exit(self._ctx)

    @property
    def devh(self):
        return self._devh

    def get_u32(self, unit: int, cid: int) -> int:
        raw = _cci_get(self._devh, unit, cid, 4)
        return struct.unpack("<I", raw)[0]

    def set_u32(self, unit: int, cid: int, value: int):
        _cci_set(self._devh, unit, cid, struct.pack("<I", value))

    def run_cmd(self, unit: int, cid: int):
        _cci_run(self._devh, unit, cid)

    # convenience -----------------------------------------------------------

    def read_all_key_filters(self) -> dict:
        """Read the enable/state of the most important filters."""
        results = {}
        queries = [
            ("AGC Enable State",             AGC_UNIT_ID, 1),
            ("AGC Policy (0=Lin 1=HEQ)",     AGC_UNIT_ID, 2),
            ("AGC Calc Enable",              AGC_UNIT_ID, 19),
            ("Radiometry Enable",            RAD_UNIT_ID, 5),
            ("TLinear Enable",               RAD_UNIT_ID, 37),
            ("TLinear Resolution (0=0.1K)",  RAD_UNIT_ID, 38),
            ("Radiometry Filter",            RAD_UNIT_ID, 23),
            ("CNF Scale Factor",             RAD_UNIT_ID, 34),
            ("TNF Scale Factor",             RAD_UNIT_ID, 35),
            ("SNF Scale Factor",             RAD_UNIT_ID, 36),
            ("Bad Pixel Replace",            OEM_UNIT_ID, 27),
            ("Temporal Filter (OEM)",        OEM_UNIT_ID, 28),
            ("Column Noise Filter (OEM)",    OEM_UNIT_ID, 29),
            ("Pixel Noise Filter (OEM)",     OEM_UNIT_ID, 30),
            ("Video Gamma Enable",           OEM_UNIT_ID, 13),
            ("Telemetry Enable",             SYS_UNIT_ID, 7),
            ("Gain Mode (0=Hi 1=Lo 2=Auto)", SYS_UNIT_ID, 17),
            ("Polarity (0=WH 1=BH)",        VID_UNIT_ID, 1),
            ("LUT / Palette Select",         VID_UNIT_ID, 2),
            ("SBNUC Enable",                 VID_UNIT_ID, 8),
            ("Freeze Enable",                VID_UNIT_ID, 10),
            ("Focus Calc Enable",            VID_UNIT_ID, 4),
        ]
        for label, unit, cid in queries:
            try:
                val = self.get_u32(unit, cid)
                results[label] = val
            except Exception as e:
                results[label] = f"error: {e}"
        return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_catalogue():
    """Print the full filter reference table."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FLIR Lepton — Complete On-Board Filter / Control Reference")
    print(f"{sep}\n")

    total = 0
    for mod_name, mod in FILTER_CATALOGUE.items():
        print(f"  ── {mod_name} Module (UVC Unit {mod['unit']}) "
              f"{'─' * (48 - len(mod_name))}")
        for name, ctrl in mod["controls"].items():
            rw = ctrl["rw"].upper()
            total += 1
            print(f"    [{total:2d}] {name:<38s} {rw:<4s}  {ctrl['desc']}")
        print()

    print(f"  Total controllable items: {total}")
    print(f"\n  Key on/off filters (shortcut names for --set):")
    for short, (mod, name) in sorted(SHORTCUT_MAP.items()):
        print(f"    {short:<22s} → {mod} / {name}")
    print(sep)


def print_states(states: dict, header: str):
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {header}")
    print(sep)
    for k, v in states.items():
        if isinstance(v, int):
            print(f"    {k:<38s} : {v}  (0x{v:X})")
        else:
            print(f"    {k:<38s} : {v}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Query and control FLIR Lepton on-board filters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python lepton_filters.py                        # telemetry mode (read-only)
  python lepton_filters.py --catalogue            # print full filter reference
  python lepton_filters.py --libuvc               # read all filters via CCI
  python lepton_filters.py --libuvc --set agc on  # enable AGC
  python lepton_filters.py --libuvc --set agc off # disable AGC
  python lepton_filters.py --libuvc --set tlinear on
  python lepton_filters.py --libuvc --set gain_mode auto
  python lepton_filters.py --libuvc --run ffc     # trigger FFC
""")
    parser.add_argument("--device", type=int, default=None,
                        help="Video device index for telemetry mode")
    parser.add_argument("--libuvc", action="store_true",
                        help="Use libuvc for full CCI get/set access")
    parser.add_argument("--catalogue", action="store_true",
                        help="Print the full filter reference table and exit")
    parser.add_argument("--set", nargs=2, metavar=("FILTER", "VALUE"),
                        help="Set a filter (requires --libuvc)")
    parser.add_argument("--run", choices=["ffc"],
                        help="Run a command (requires --libuvc)")
    args = parser.parse_args()

    # -- catalogue only --
    if args.catalogue:
        print_catalogue()
        return

    # -- libuvc mode --
    if args.libuvc:
        try:
            with LeptonCCI() as cci:
                if args.set:
                    name_key, raw_val = args.set
                    name_key = name_key.lower()
                    if name_key not in SHORTCUT_MAP:
                        print(f"Unknown filter shortcut '{name_key}'.")
                        print("Run with --catalogue to see available shortcuts.")
                        sys.exit(1)

                    mod_name, ctrl_name = SHORTCUT_MAP[name_key]
                    ctrl = FILTER_CATALOGUE[mod_name]["controls"][ctrl_name]
                    unit = FILTER_CATALOGUE[mod_name]["unit"]

                    val_str = raw_val.lower()
                    if val_str in VALUE_ALIASES:
                        value = VALUE_ALIASES[val_str]
                    else:
                        try:
                            value = int(raw_val, 0)
                        except ValueError:
                            print(f"Cannot parse value '{raw_val}'.")
                            sys.exit(1)

                    print(f"Setting {mod_name}/{ctrl_name} (unit={unit}, cid={ctrl['cid']}) → {value}")
                    cci.set_u32(unit, ctrl["cid"], value)

                    readback = cci.get_u32(unit, ctrl["cid"])
                    print(f"Readback: {readback}  ({'OK' if readback == value else 'MISMATCH'})")

                elif args.run == "ffc":
                    print("Triggering Flat-Field Correction (FFC)...")
                    cci.run_cmd(RAD_UNIT_ID, 12)
                    print("FFC command sent.")

                else:
                    states = cci.read_all_key_filters()
                    print_states(states, "Lepton Filter States (via CCI / libuvc)")

        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        return

    # -- telemetry mode (default) --
    print("Reading filter states from telemetry (flirpy)...")
    print("(For full get/set control, use --libuvc)\n")
    try:
        states = read_filter_states_from_telemetry(args.device)
        print_states(states, "Lepton Filter States (from telemetry)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\nTip: run with --catalogue to see all 50+ controllable filters.")


if __name__ == "__main__":
    main()
