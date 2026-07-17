"""
Thermal Camera Recording Script

Records thermal video data from one or more FLIR Boson cameras, with optional
compression and configurable duration.

Usage:
    # Single camera, auto-detected
    python record_thermal_video.py --output recording.npz --duration 30

    # Multiple cameras, explicit device/port
    python record_thermal_video.py --camera 1:COM4 --camera 2:COM6 \\
        --output dual_recording.npz --duration 60 --compress
"""

import sys
import argparse
import time
from typing import List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from wrapper_boson import BosonWithTelemetry


def parse_camera_spec(spec: str) -> Tuple[int, Optional[str]]:
    """
    Parse a "--camera" value of the form "DEVICE" or "DEVICE:PORT".

    Args:
        spec: Raw command line value, e.g. "0" or "1:COM4"

    Returns:
        Tuple of (device index, port or None)
    """
    if ":" in spec:
        device_str, port = spec.rsplit(":", 1)
    else:
        device_str, port = spec, None
    return int(device_str), port


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record thermal video from one or more FLIR Boson cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record 30 seconds from an auto-detected single camera
    python record_thermal_video.py --output thermal_data.npz --duration 30

    # Record with compression
    python record_thermal_video.py --output compressed_data.npz --duration 60 --compress

    # Record from two cameras (device 1 on COM4, device 2 on COM6)
    python record_thermal_video.py --camera 1:COM4 --camera 2:COM6 \\
        --output dual_recording.npz --duration 120
        """
    )
    parser.add_argument('--camera', action='append', dest='cameras', default=None,
                       help='Camera as DEVICE or DEVICE:PORT (repeatable). '
                            'If omitted, a single auto-detected camera is used.')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (will create directories if needed)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration of recording in seconds, -1 for manual stop (default: 10)')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Temporal downsample factor for frames (default: 1, no downsampling)')
    parser.add_argument('--compress', action='store_true',
                       help='Compress output using zstandard compression')
    parser.add_argument('--disable-auto-ffc', action='store_true',
                       help='Disable automatic FFC for the recording session')
    parser.add_argument('--force-ffc-at-init', action='store_true',
                       help='Force FFC at initialization')
    parser.add_argument('--leave-ffc-disabled', action='store_true',
                       help='Leave FFC disabled after recording')
    return parser.parse_args()


def initialize_cameras(camera_specs: List[str], disable_auto_ffc: bool,
                      force_ffc_at_init: bool) -> List[BosonWithTelemetry]:
    """
    Initialize and configure one or more thermal cameras.

    Args:
        camera_specs: Raw "--camera" values, or empty for a single auto-detected camera
        disable_auto_ffc: Whether to disable automatic FFC during recording
        force_ffc_at_init: Whether to force FFC at initialization

    Returns:
        List of configured camera objects

    Raises:
        RuntimeError: If any camera fails to initialize
    """
    specs = [parse_camera_spec(s) for s in camera_specs] if camera_specs else [(None, None)]
    cameras: List[BosonWithTelemetry] = []

    try:
        for i, (device, port) in enumerate(specs):
            print(f"Connecting to thermal camera {i} (device={device}, port={port})...")
            camera = BosonWithTelemetry(device=device, port=port)
            cameras.append(camera)
            print(f"Camera {i} connected successfully.")

        if force_ffc_at_init:
            for camera in cameras:
                camera.camera.do_ffc()
            time.sleep(1)  # Allow FFC to complete
            print("Forced FFC performed at initialization.")

        if disable_auto_ffc:
            for camera in cameras:
                camera.camera.set_ffc_manual()

        return cameras

    except Exception as e:
        print(f"Failed to connect to thermal camera(s): {e}")
        print("Please check:")
        print("- Camera(s) are properly connected via USB")
        print("- Camera drivers are installed")
        print("- No other applications are using the camera(s)")
        for camera in cameras:
            camera.stop()
            camera.close()
        raise RuntimeError("Camera initialization failed") from e


def record_thermal_data(cameras: List[BosonWithTelemetry], duration: int, downsample: int) -> bool:
    """
    Record synchronized thermal data from all cameras.

    Args:
        cameras: Initialized camera objects
        duration: Recording duration in seconds (-1 for manual stop)
        downsample: Temporal downsample factor for frames

    Returns:
        True if recording completed successfully, False if cancelled by user
    """
    print(f"\nPreparing to record from {len(cameras)} camera(s)...")
    if duration > 0:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Until manually stopped")

    print("Press Enter to start recording, or 'q' to cancel...")
    user_input = input().strip().lower()
    if user_input == 'q':
        print("Recording cancelled by user.")
        return False

    for camera in cameras:
        camera.set_downsample_factor(downsample)

    # Small placeholder figure used purely to detect a keypress to stop early.
    # Created *before* start_logging() so its window-setup cost (a few hundred
    # ms of GIL-heavy work) doesn't compete with the capture thread right at
    # the start of the logging window and show up as spurious dropped frames.
    fig = plt.figure(figsize=(1, 1))
    plt.axis('off')
    plt.title('Recording... press any key to stop')
    plt.show(block=False)
    plt.pause(0.001)

    target_duration = duration if duration > 0 else float('inf')
    print("Recording started. Press any key in the popup window to stop early...")

    start_time = time.time()
    for camera in cameras:
        camera.start_logging()

    try:
        with tqdm(total=duration if duration > 0 else None, desc="Recording", unit="s") as pbar:
            while time.time() - start_time < target_duration:
                elapsed = time.time() - start_time
                if duration > 0:
                    pbar.update(elapsed - pbar.n)
                else:
                    pbar.update(elapsed - pbar.n if pbar.n < elapsed else 0)

                if plt.waitforbuttonpress(timeout=0.1):
                    print("\nRecording stopped early by user.")
                    break

        return True

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        return True  # Still save partial data

    finally:
        for camera in cameras:
            camera.stop_logging()
        plt.close(fig)


def save_data(cameras: List[BosonWithTelemetry], output_file: str, compress: bool) -> None:
    """
    Save recorded thermal data to file.

    Args:
        cameras: Camera objects containing recorded data
        output_file: Output file path
        compress: Whether to compress the data
    """
    print("Processing and saving data...")

    arrays = {}
    if len(cameras) == 1:
        camera = cameras[0]
        arrays['raw_thr_frames'] = np.array(camera.logged_images)
        arrays['raw_thr_tstamps'] = np.array(camera.logged_tstamps)
        arrays['thr_cam_timestamp_offset'] = camera.timestamp_offset
        arrays['dropped_frame_count'] = camera.dropped_frame_count
        print(f"Captured {len(arrays['raw_thr_frames'])} frames "
              f"({camera.dropped_frame_count} dropped)")
    else:
        for i, camera in enumerate(cameras):
            frames = np.array(camera.logged_images)
            arrays[f'raw_thr_frames_{i}'] = frames
            arrays[f'raw_thr_tstamps_{i}'] = np.array(camera.logged_tstamps)
            arrays[f'thr_cam_timestamp_offset_{i}'] = camera.timestamp_offset
            arrays[f'dropped_frame_count_{i}'] = camera.dropped_frame_count
            print(f"Camera {i} captured {len(frames)} frames "
                  f"({camera.dropped_frame_count} dropped)")

    np.savez(output_file, **arrays)
    print(f"Data saved to: {output_file}")

    if compress:
        import subprocess
        compressed_file = output_file + ".zst"
        print("Compressing with zstd...")
        try:
            subprocess.run(["zstd", "-f", output_file, "-o", compressed_file], check=True)
            print(f"Compressed file saved to: {compressed_file}")
            output_file = compressed_file
            Path(output_file[:-4]).unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Compression failed: {e}")
        except FileNotFoundError:
            print("Warning: zstd command not found. Install zstd to enable compression.")

    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")


def cleanup_cameras(cameras: List[BosonWithTelemetry], leave_ffc_disabled: bool = False) -> None:
    """
    Properly cleanup camera resources.

    Args:
        cameras: Camera objects to cleanup
        leave_ffc_disabled: Whether to leave FFC disabled after recording
    """
    for i, camera in enumerate(cameras):
        try:
            if not leave_ffc_disabled:
                camera.camera.set_ffc_auto()
                print(f"Camera {i} FFC reset to automatic mode.")
            camera.stop()
            camera.close()
            print(f"Camera {i} resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error during camera {i} cleanup: {e}")


def main() -> None:
    """Main function to orchestrate the recording process."""
    args = parse_args()

    if args.downsample < 1:
        print("Error: Downsample factor must be >= 1")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    cameras: List[BosonWithTelemetry] = []
    try:
        cameras = initialize_cameras(args.cameras, args.disable_auto_ffc, args.force_ffc_at_init)

        if record_thermal_data(cameras, args.duration, args.downsample):
            save_data(cameras, output_file, args.compress)
            print("Recording completed successfully!")
        else:
            print("Recording was cancelled.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    finally:
        cleanup_cameras(cameras, args.leave_ffc_disabled)
        plt.close('all')


if __name__ == "__main__":
    main()
