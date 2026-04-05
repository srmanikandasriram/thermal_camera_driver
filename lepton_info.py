"""
FLIR Lepton Camera Info Tool

Connects to a FLIR Lepton camera via PureThermal board, grabs a frame
with telemetry, and prints out the camera serial number, model, firmware
revision, and real-time diagnostics.

Requires: flirpy, opencv-python, numpy

Usage:
    python lepton_info.py
    python lepton_info.py --device 0
"""

import argparse
import sys
import struct
import time
import numpy as np

from flirpy.camera.lepton import Lepton


def parse_telemetry_row(raw_row: np.ndarray) -> dict:
    """
    Parse a full Lepton telemetry row (Row A) from raw uint16 pixel data.

    The telemetry layout follows the FLIR Lepton Software IDD (110-0144):
      Word  0      : Telemetry Revision
      Words 1-2    : Time Counter (ms, 32-bit)
      Words 3-4    : Status Bits (32-bit)
      Words 5-6    : Module Serial Number (upper 32 bits)
      Words 7-8    : Module Serial Number (lower 32 bits)
      Words 9-10   : Software Revision (upper 32 bits)
      Words 11-12  : Software Revision (lower 32 bits)
      Words 13-15  : Reserved
      Words 16-17  : Frame Counter (32-bit)
      Word  18     : Frame Mean
      Word  19     : FPA Temperature (Kelvin × 100)
      Word  20     : Housing Temperature (Kelvin × 100)
      Words 21-24  : FPA Temperature at last FFC (19) + Time since last FFC ...
      Word  26     : AGC ROI top
      Word  27     : AGC ROI left
      Word  28     : AGC ROI bottom
      Word  29     : AGC ROI right
      Word  30     : AGC Clip Limit High
      Word  31     : AGC Clip Limit Low
      Words 32-33  : Video Output Format (32-bit)
    """
    words = raw_row.view(np.uint16).copy()

    info = {}
    info["telemetry_revision"] = int(words[0])
    info["time_counter_ms"] = int(words[1]) << 16 | int(words[2])

    status_bits = int(words[3]) << 16 | int(words[4])
    info["status_raw"] = status_bits
    info["ffc_desired"] = bool(status_bits & (1 << 0))
    info["ffc_state"] = "In Progress" if (status_bits & (1 << 1)) else "Idle"
    info["agc_enabled"] = bool(status_bits & (1 << 4))

    serial_upper = int(words[5]) << 16 | int(words[6])
    serial_lower = int(words[7]) << 16 | int(words[8])
    info["serial_number"] = (serial_upper << 32) | serial_lower

    sw_upper = int(words[9]) << 16 | int(words[10])
    sw_lower = int(words[11]) << 16 | int(words[12])
    gpp_major = (sw_upper >> 24) & 0xFF
    gpp_minor = (sw_upper >> 16) & 0xFF
    gpp_build = sw_upper & 0xFFFF
    dsp_major = (sw_lower >> 24) & 0xFF
    dsp_minor = (sw_lower >> 16) & 0xFF
    dsp_build = sw_lower & 0xFFFF
    info["software_rev_gpp"] = f"{gpp_major}.{gpp_minor}.{gpp_build}"
    info["software_rev_dsp"] = f"{dsp_major}.{dsp_minor}.{dsp_build}"

    info["frame_counter"] = int(words[16]) << 16 | int(words[17])
    info["frame_mean"] = int(words[18])

    fpa_raw = int(words[19])
    housing_raw = int(words[20])
    info["fpa_temp_kelvin"] = fpa_raw / 100.0 if fpa_raw else None
    info["fpa_temp_celsius"] = (fpa_raw / 100.0 - 273.15) if fpa_raw else None
    info["housing_temp_kelvin"] = housing_raw / 100.0 if housing_raw else None
    info["housing_temp_celsius"] = (housing_raw / 100.0 - 273.15) if housing_raw else None

    ffc_fpa_raw = int(words[21])
    info["fpa_temp_at_last_ffc_celsius"] = (ffc_fpa_raw / 100.0 - 273.15) if ffc_fpa_raw else None

    ffc_elapsed = int(words[22]) << 16 | int(words[23])
    info["time_since_last_ffc_ms"] = ffc_elapsed

    if len(words) > 33:
        info["agc_roi"] = (int(words[26]), int(words[27]),
                           int(words[28]), int(words[29]))
        info["agc_clip_high"] = int(words[30])
        info["agc_clip_low"] = int(words[31])
        video_fmt = int(words[32]) << 16 | int(words[33])
        info["video_output_format_raw"] = video_fmt

    return info


def identify_model(width: int, height: int) -> str:
    if width == 160 and height == 120:
        return "Lepton 3 / 3.5  (160×120)"
    elif width == 80 and height == 60:
        return "Lepton 2 / 2.5  (80×60)"
    else:
        return f"Unknown ({width}×{height})"


def print_camera_info(info: dict, img_width: int, img_height: int,
                      full_height: int) -> None:
    sep = "=" * 56
    print(f"\n{sep}")
    print("       FLIR Lepton Camera Information")
    print(sep)

    print(f"\n  Model (by resolution) : {identify_model(img_width, img_height)}")
    print(f"  Thermal resolution    : {img_width} × {img_height}")
    print(f"  Frame w/ telemetry    : {img_width} × {full_height}")

    serial = info.get("serial_number", 0)
    if serial:
        print(f"\n  Serial Number         : {serial}")
        print(f"  Serial Number (hex)   : 0x{serial:016X}")
    else:
        print(f"\n  Serial Number         : Not available (0)")

    print(f"\n  Firmware (GPP)        : {info.get('software_rev_gpp', 'N/A')}")
    print(f"  Firmware (DSP)        : {info.get('software_rev_dsp', 'N/A')}")

    rev = info.get("telemetry_revision", 0)
    print(f"  Telemetry Revision    : {rev}  (0x{rev:04X})")

    print(f"\n  Uptime                : {info.get('time_counter_ms', 0)} ms"
          f"  ({info.get('time_counter_ms', 0) / 1000:.1f} s)")
    print(f"  Frame Counter         : {info.get('frame_counter', 0)}")
    print(f"  Frame Mean Value      : {info.get('frame_mean', 0)}")

    fpa_c = info.get("fpa_temp_celsius")
    fpa_k = info.get("fpa_temp_kelvin")
    if fpa_c is not None:
        print(f"\n  FPA Temperature       : {fpa_c:.2f} °C  ({fpa_k:.2f} K)")
    else:
        print(f"\n  FPA Temperature       : N/A")

    housing_c = info.get("housing_temp_celsius")
    housing_k = info.get("housing_temp_kelvin")
    if housing_c is not None:
        print(f"  Housing Temperature   : {housing_c:.2f} °C  ({housing_k:.2f} K)")
    else:
        print(f"  Housing Temperature   : N/A")

    print(f"\n  FFC State             : {info.get('ffc_state', 'N/A')}")
    print(f"  FFC Desired           : {info.get('ffc_desired', 'N/A')}")
    ffc_fpa = info.get("fpa_temp_at_last_ffc_celsius")
    if ffc_fpa is not None:
        print(f"  FPA Temp at last FFC  : {ffc_fpa:.2f} °C")
    ffc_elapsed = info.get("time_since_last_ffc_ms", 0)
    print(f"  Time since last FFC   : {ffc_elapsed} ms  ({ffc_elapsed / 1000:.1f} s)")

    print(f"\n  AGC Enabled           : {info.get('agc_enabled', 'N/A')}")
    agc_roi = info.get("agc_roi")
    if agc_roi:
        print(f"  AGC ROI (T,L,B,R)    : {agc_roi}")
    clip_h = info.get("agc_clip_high")
    clip_l = info.get("agc_clip_low")
    if clip_h is not None:
        print(f"  AGC Clip High/Low     : {clip_h} / {clip_l}")

    print(f"\n  Status Register       : 0x{info.get('status_raw', 0):08X}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Print FLIR Lepton camera serial number and specifications.")
    parser.add_argument("--device", type=int, default=None,
                        help="Video device index (default: auto-detect)")
    args = parser.parse_args()

    print("Connecting to FLIR Lepton camera...")

    cam = Lepton()

    try:
        cam.setup_video(args.device)
    except Exception as e:
        print(f"ERROR: Could not open camera — {e}")
        print("Make sure the PureThermal board is plugged in via USB.")
        sys.exit(1)

    print("Camera opened. Grabbing frames to read telemetry...")

    good_frame = None
    for attempt in range(30):
        raw = cam.grab(strip_telemetry=False)
        if raw is not None and raw.size > 0:
            good_frame = raw
            if attempt >= 2:
                break
        time.sleep(0.15)

    if good_frame is None:
        print("ERROR: Failed to capture any frames. Camera may not be responding.")
        cam.release()
        sys.exit(1)

    full_height, full_width = good_frame.shape[:2]
    img_height = full_height - 2
    img_width = full_width

    telemetry_row = good_frame[-2, :]
    info = parse_telemetry_row(telemetry_row)

    print_camera_info(info, img_width, img_height, full_height)

    cam.release()
    print("\nCamera released. Done.")


if __name__ == "__main__":
    main()
