"""
Boson + Lepton Simultaneous Thermal Camera Recording Script

Records synchronized thermal video data from a FLIR Boson and a FLIR Lepton
camera simultaneously, preserving full telemetry for both.

The output is two separate .npz files (one per camera), named
``<stem>_boson.npz`` and ``<stem>_lepton.npz``.  Each file uses the
same key names as the corresponding single-camera recording script so
that analyze_data.py can load them without modification.

Usage:
    python record_boson_lepton_video.py --output recording.npz --duration 60
    python record_boson_lepton_video.py --output data.npz --duration 120 --compress

Author: [Your Name]
Date: April 2026
"""

import sys
import argparse
import time
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from wrapper_boson import BosonWithTelemetry
from wrapper_lepton import LeptonWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record thermal video from a FLIR Boson and a FLIR Lepton simultaneously.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record for 2 minutes
    python record_boson_lepton_video.py --output experiment_001.npz --duration 120

    # Record until manually stopped
    python record_boson_lepton_video.py --output continuous.npz

    # Record with compression
    python record_boson_lepton_video.py --output data.npz --duration 60 --compress

Note:
    - Boson device index and serial port can be set with --boson-device / --boson-port
    - Lepton is auto-detected via flirpy
    - Press any key in the matplotlib window to stop early
        """,
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output file name (stored in ./boson_lepton_data/ directory)",
    )
    parser.add_argument(
        "--duration", type=int, default=-1,
        help="Duration in seconds. Use -1 for manual stop (default: -1)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Compress output using zstandard compression",
    )
    parser.add_argument(
        "--boson-device", type=int, default=0,
        help="Boson video device index (default: 1)",
    )
    parser.add_argument(
        "--boson-port", type=str, default="COM3",
        help="Boson serial port (default: COM4)",
    )
    return parser.parse_args()


def initialize_cameras(
    boson_device: int = 1,
    boson_port: str = "COM4",
) -> Tuple[Optional[BosonWithTelemetry], Optional[LeptonWrapper]]:
    """
    Initialize the Boson and Lepton cameras.

    Args:
        boson_device: Boson video device index
        boson_port: Boson serial port

    Returns:
        (boson, lepton) or (None, None) if either camera fails.
    """
    boson = None
    lepton = None

    # --- Boson ---
    try:
        print(f"Connecting to FLIR Boson (device {boson_device}, {boson_port})...")
        boson = BosonWithTelemetry(device=boson_device, port=boson_port)
        print("Boson connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Boson: {e}")
        print("Please check:")
        print(f"  - Camera is connected and visible as device {boson_device}")
        print(f"  - Serial port {boson_port} is correct (use --boson-port to change)")
        print("  - No other application is using the camera")
        return None, None

    # --- Lepton ---
    try:
        print("Connecting to FLIR Lepton...")
        lepton = LeptonWrapper()

        print("Waiting for first Lepton frame...", end="", flush=True)
        for _ in range(50):
            with lepton._lock:
                ready = lepton._latest_frame is not None
            if ready:
                break
            time.sleep(0.1)
        else:
            print(" timed out.")
            print("Lepton did not produce a frame within 5 seconds.")
            boson.stop()
            if getattr(boson, "camera", None):
                boson.camera.close()
            lepton.close()
            return None, None

        print(" OK")
        print("Lepton connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Lepton: {e}")
        print("Please check:")
        print("  - Lepton / PureThermal board is connected via USB")
        print("  - flirpy is installed: pip install flirpy")
        print("  - No other application is using the camera")
        if boson:
            boson.stop()
            if getattr(boson, "camera", None):
                boson.camera.close()
        return None, None

    # --- Configure Boson FFC (Lepton handles FFC automatically) ---
    print("Performing flat field correction (FFC) on Boson...")
    boson.camera.do_ffc()
    boson.camera.set_ffc_manual()
    time.sleep(1)

    print("Both cameras connected and configured.")
    return boson, lepton


def record_thermal_data(
    boson: BosonWithTelemetry,
    lepton: LeptonWrapper,
    duration: int,
) -> bool:
    """
    Record synchronized thermal data from both cameras.

    Args:
        boson: Boson camera object
        lepton: Lepton camera object
        duration: Recording duration in seconds (-1 for manual stop)

    Returns:
        True if recording completed (fully or partially), False if cancelled
    """
    print(f"\nPreparing to record from both cameras...")
    if duration > 0:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Until manually stopped")

    print("Press Enter to start recording, or 'q' to cancel...")
    user_input = input().strip().lower()
    if user_input == "q":
        print("Recording cancelled by user.")
        return False

    try:
        start_time = time.time()
        boson.start_logging()
        lepton.start_logging()

        target_duration = duration if duration > 0 else float("inf")

        if duration == -1:
            print("Recording started. Press any key to stop...")
        else:
            print(f"Recording started for {duration} seconds. Press any key to stop early...")

        fig = plt.figure(figsize=(1, 1))
        plt.axis("off")
        plt.title("Recording... Press any key to stop")
        plt.show(block=False)

        if duration > 0:
            with tqdm(total=duration, desc="Recording", unit="s") as pbar:
                while time.time() - start_time < target_duration:
                    elapsed = time.time() - start_time
                    pbar.update(elapsed - pbar.n)
                    if plt.waitforbuttonpress(timeout=0.1):
                        print("\nRecording stopped early by user.")
                        break
                    time.sleep(0.1)
        else:
            while time.time() - start_time < target_duration:
                if plt.waitforbuttonpress(timeout=0.1):
                    elapsed = time.time() - start_time
                    print(f"\nRecording stopped after {elapsed:.1f} seconds.")
                    break
                time.sleep(0.1)

        plt.close(fig)
        return True

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        plt.close("all")
        return True

    finally:
        boson.stop_logging()
        lepton.stop_logging()


def _compress_file(path: str) -> str:
    """Compress *path* with zstd, remove the original, and return the final path."""
    import subprocess
    compressed = path + ".zst"
    print(f"  Compressing {Path(path).name}...")
    try:
        subprocess.run(["zstd", "-f", path, "-o", compressed], check=True)
        Path(path).unlink(missing_ok=True)
        return compressed
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Compression failed: {e}")
    except FileNotFoundError:
        print("  Warning: zstd command not found. Install zstd to enable compression.")
    return path


def save_data(
    boson: BosonWithTelemetry,
    lepton: LeptonWrapper,
    output_file: str,
    compress: bool = False,
) -> None:
    """
    Save recorded data as two separate .npz files, one per camera.

    File names are derived from *output_file* by inserting ``_boson`` /
    ``_lepton`` before the ``.npz`` extension.  Each file uses the same
    key names as the corresponding single-camera recording script so
    that ``analyze_data.py`` can load them without modification.
    """
    print("Processing and saving data...")

    base = Path(output_file)
    stem = base.stem
    suffix = base.suffix  # .npz
    parent = base.parent

    boson_file = str(parent / f"{stem}_boson{suffix}")
    lepton_file = str(parent / f"{stem}_lepton{suffix}")

    # -- Boson --
    raw_frames_boson = np.array(boson.logged_images)
    timestamps_boson = np.array(boson.logged_tstamps)
    cam_tstamps_boson = np.array(boson.logged_cam_tstamps)
    frame_numbers_boson = np.array(boson.logged_frame_numbers)
    offset_boson = boson.timestamp_offset

    # ThreadedBoson callbacks may deliver frames with an extra trailing
    # channel dimension (e.g. H×W×1).  Squeeze it away but keep the
    # telemetry header rows (first 2 rows of each 514-row frame) intact.
    if raw_frames_boson.ndim == 4 and raw_frames_boson.shape[-1] == 1:
        raw_frames_boson = raw_frames_boson[:, :, :, 0]

    print(f"Boson  captured {len(raw_frames_boson)} frames"
          f"  (shape per frame: {raw_frames_boson.shape[1:]})")

    np.savez(
        boson_file,
        raw_thr_frames=raw_frames_boson,
        raw_thr_tstamps=timestamps_boson,
        thr_cam_timestamp_offset=offset_boson,
        raw_thr_cam_tstamps=cam_tstamps_boson,
        raw_thr_frame_numbers=frame_numbers_boson,
    )
    print(f"  Boson data saved to: {boson_file}")

    # -- Lepton --
    raw_frames_lepton = (np.array(lepton.logged_images)
                         if lepton.logged_images else np.empty(0))
    timestamps_lepton = np.array(lepton.logged_tstamps)
    cam_tstamps_lepton = np.array(lepton.logged_cam_tstamps)
    frame_numbers_lepton = np.array(lepton.logged_frame_numbers)
    offset_lepton = lepton.timestamp_offset

    print(f"Lepton captured {len(raw_frames_lepton)} frames")

    if len(raw_frames_lepton) > 0:
        thermal_h = raw_frames_lepton.shape[1] - 2
        print(f"  Lepton thermal: {raw_frames_lepton.shape[2]}x{thermal_h}, "
              f"with telemetry: {raw_frames_lepton.shape[2]}x{raw_frames_lepton.shape[1]}")

    np.savez(
        lepton_file,
        raw_thr_frames=raw_frames_lepton,
        raw_thr_tstamps=timestamps_lepton,
        thr_cam_timestamp_offset=offset_lepton,
        raw_thr_cam_tstamps=cam_tstamps_lepton,
        raw_thr_frame_numbers=frame_numbers_lepton,
    )
    print(f"  Lepton data saved to: {lepton_file}")

    # -- Optional compression --
    if compress:
        boson_file = _compress_file(boson_file)
        lepton_file = _compress_file(lepton_file)

    # -- Summary --
    boson_sz = Path(boson_file).stat().st_size / (1024 * 1024)
    lepton_sz = Path(lepton_file).stat().st_size / (1024 * 1024)
    print(f"File sizes: Boson {boson_sz:.1f} MB, Lepton {lepton_sz:.1f} MB")

    if len(timestamps_boson) > 0 and len(timestamps_lepton) > 0:
        time_diff = abs(timestamps_boson[0] - timestamps_lepton[0])
        print(f"Camera synchronization offset: {time_diff * 1000:.1f} ms")


def cleanup_cameras(
    boson: Optional[BosonWithTelemetry],
    lepton: Optional[LeptonWrapper],
) -> None:
    if boson:
        try:
            boson.stop()
            boson.camera.close()
            print("Boson resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error cleaning up Boson: {e}")
    if lepton:
        try:
            lepton.close()
            print("Lepton resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error cleaning up Lepton: {e}")


def main() -> None:
    args = parse_args()

    output_path = Path("./boson_lepton_data") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    boson = None
    lepton = None

    try:
        boson, lepton = initialize_cameras(args.boson_device, args.boson_port)
        if not boson or not lepton:
            print("Failed to initialize both cameras. Exiting.")
            sys.exit(1)

        if record_thermal_data(boson, lepton, args.duration):
            save_data(boson, lepton, output_file, args.compress)
            print("Boson + Lepton recording completed successfully!")
        else:
            print("Recording was cancelled.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    finally:
        cleanup_cameras(boson, lepton)
        plt.close("all")


if __name__ == "__main__":
    main()
