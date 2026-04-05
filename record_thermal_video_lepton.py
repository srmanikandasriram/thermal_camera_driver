"""
FLIR Lepton Thermal Camera Recording Script

Records thermal video data from a FLIR Lepton camera and saves it as a
NumPy archive (.npz), optionally compressed with zstandard.

The output format is identical to record_thermal_video.py (Boson), so
analyze_data.py can load Lepton recordings without modification.

Usage:
    python record_thermal_video_lepton.py --output recording.npz --duration 30
    python record_thermal_video_lepton.py --output data.npz --duration 60 --compress

Author: [Your Name]
Date: March 2026
"""

import sys
import argparse
import time
from typing import Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path

from wrapper_lepton import LeptonWrapper


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Record thermal video from a FLIR Lepton camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record 30 seconds to file
    python record_thermal_video_lepton.py --output lepton_data.npz --duration 30

    # Record with compression
    python record_thermal_video_lepton.py --output lepton_data.npz --duration 60 --compress

    # Record to specific directory
    python record_thermal_video_lepton.py --output lepton_recordings/exp_001.npz --duration 120
        """,
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output file path (directories will be created if needed)",
    )
    parser.add_argument(
        "--duration", type=int, default=10,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Compress output using zstandard compression",
    )
    return parser.parse_args()


def initialize_camera() -> Optional[LeptonWrapper]:
    """
    Initialize and connect to the Lepton camera.

    Returns:
        Configured LeptonWrapper object, or None if initialization fails
    """
    try:
        print("Connecting to FLIR Lepton thermal camera...")
        camera = LeptonWrapper()

        # Wait for the first frame so we know the camera is producing data
        print("Waiting for first frame...", end="", flush=True)
        for _ in range(50):
            import threading as _threading
            with camera._lock:
                ready = camera._latest_frame is not None
            if ready:
                break
            time.sleep(0.1)
        else:
            print(" timed out.")
            print("Camera did not produce a frame within 5 seconds.")
            camera.close()
            return None

        print(" OK")
        print("Camera connected successfully.")
        print("Note: Lepton handles FFC (flat field correction) automatically.")
        return camera

    except Exception as e:
        print(f"Failed to connect to Lepton camera: {e}")
        print("Please check:")
        print("  - Camera is properly connected via USB")
        print("  - flirpy is installed: pip install flirpy")
        print("  - No other application is using the camera")
        return None


def record_thermal_data(camera: LeptonWrapper, duration: int) -> bool:
    """
    Record thermal data for the specified duration.

    Args:
        camera: Initialized LeptonWrapper object
        duration: Recording duration in seconds

    Returns:
        True if recording completed (fully or partially), False if cancelled
    """
    print(f"\nPreparing to record for {duration} seconds...")
    print("Press Enter to start recording, or 'q' to cancel...")

    user_input = input().strip().lower()
    if user_input == "q":
        print("Recording cancelled by user.")
        return False

    print("Recording started. Press Ctrl+C to stop early.")
    start_time = time.time()
    camera.start_logging()

    try:
        with tqdm(total=duration, desc="Recording", unit="s") as pbar:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                pbar.update(elapsed - pbar.n)
                time.sleep(0.1)

        return True

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        return True  # Still save whatever was captured

    finally:
        camera.stop_logging()


def save_data(camera: LeptonWrapper, output_file: str, compress: bool) -> None:
    """
    Save recorded thermal data to a .npz file.

    The core key names (raw_thr_frames, raw_thr_tstamps,
    thr_cam_timestamp_offset) match the Boson recording output so that
    analyze_data.py works for both cameras.  Two additional Lepton-specific
    arrays provide camera-clock timestamps and frame counters extracted from
    the embedded telemetry footer.

    Frames are saved WITH their 2-row telemetry footer (e.g. 122 rows for
    Lepton 3).  Strip the last 2 rows to get the thermal-only image, or
    parse them to recover per-frame telemetry.

    Args:
        camera: LeptonWrapper containing recorded data
        output_file: Destination .npz file path
        compress: Whether to compress with zstandard after saving
    """
    if len(camera.logged_images) == 0:
        print("Warning: no frames were captured. Nothing to save.")
        return

    print("Processing and saving data...")

    raw_frames = np.array(camera.logged_images)
    timestamps = np.array(camera.logged_tstamps)
    cam_tstamps = np.array(camera.logged_cam_tstamps)
    frame_numbers = np.array(camera.logged_frame_numbers)
    timestamp_offset = camera.timestamp_offset

    thermal_h = raw_frames.shape[1] - 2
    print(f"Captured {len(raw_frames)} frames  "
          f"(thermal: {raw_frames.shape[2]}x{thermal_h}, "
          f"with telemetry: {raw_frames.shape[2]}x{raw_frames.shape[1]})")
    print(f"Timestamp offset (system − camera): {timestamp_offset:.6f} s")

    np.savez(
        output_file,
        raw_thr_frames=raw_frames,
        raw_thr_tstamps=timestamps,
        thr_cam_timestamp_offset=timestamp_offset,
        raw_thr_cam_tstamps=cam_tstamps,
        raw_thr_frame_numbers=frame_numbers,
    )
    print(f"Data saved to: {output_file}")

    if compress:
        import subprocess
        compressed_file = output_file + ".zst"
        print("Compressing with zstd...")
        try:
            subprocess.run(
                ["zstd", "-f", output_file, "-o", compressed_file], check=True
            )
            print(f"Compressed file saved to: {compressed_file}")
            Path(output_file).unlink(missing_ok=True)
            output_file = compressed_file
        except subprocess.CalledProcessError as e:
            print(f"Warning: Compression failed: {e}")
        except FileNotFoundError:
            print("Warning: zstd command not found. Install zstd to enable compression.")

    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")


def cleanup_camera(camera: Optional[LeptonWrapper]) -> None:
    """
    Release camera resources.

    Args:
        camera: LeptonWrapper to clean up (can be None)
    """
    if camera:
        try:
            camera.close()
            print("Camera resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error during camera cleanup: {e}")


def main() -> None:
    """Main function to orchestrate the recording process."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    camera = None
    try:
        camera = initialize_camera()
        if not camera:
            sys.exit(1)

        if record_thermal_data(camera, args.duration):
            save_data(camera, output_file, args.compress)
            print("Recording completed successfully!")
        else:
            print("Recording was cancelled.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    finally:
        cleanup_camera(camera)


if __name__ == "__main__":
    main()
