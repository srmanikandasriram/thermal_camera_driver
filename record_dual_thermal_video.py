"""
Dual Thermal Camera Recording Script

This script records synchronized thermal video data from two FLIR Boson cameras
simultaneously with telemetry support.

Usage:
    python record_dual_thermal_video.py --output dual_recording.npz --duration 120

Author: [Your Name]
Date: January 2026
"""

import sys
import argparse
import time
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import zstandard as zstd

from wrapper_boson import BosonWithTelemetry


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Record thermal video from two FLIR Boson cameras simultaneously.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record for 2 minutes
    python record_dual_thermal_video.py --output experiment_001.npz --duration 120
    
    # Record until manually stopped
    python record_dual_thermal_video.py --output continuous_recording.npz
    
    # Record with compression
    python record_dual_thermal_video.py --output compressed_data.npz --duration 60 --compress
    
Note:
    - Cameras are expected on device 1 (COM4) and device 2 (COM6)
    - Modify the device/port settings in the code if your setup differs
    - Press any key during recording to stop early
        """
    )
    parser.add_argument('--output', type=str, required=True,
                       help='Output file name (stored in ./dual_data/ directory)')
    parser.add_argument('--duration', type=int, default=-1,
                       help='Duration in seconds. Use -1 for manual stop (default: -1)')
    parser.add_argument('--compress', action='store_true',
                       help='Compress output using zstandard compression')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Temporal downsample factor for frames (default: 1, no downsampling)')  
    parser.add_argument('--disable-auto-ffc', action='store_true',
                       help='Disable automatic FFC for the recording session')
    parser.add_argument('--force-ffc-at-init', action='store_true',
                       help='Force FFC at initialization')
    parser.add_argument('--leave-ffc-disabled', action='store_true',
                       help='Leave FFC disabled after recording')
    return parser.parse_args()


def initialize_dual_cameras(disable_auto_ffc: bool, force_ffc_at_init: bool) -> Tuple[Optional[BosonWithTelemetry], Optional[BosonWithTelemetry]]:
    """
    Initialize both thermal cameras.
    
    Returns:
        Tuple of (camera_A, camera_B) or (None, None) if initialization fails
    """
    camera_a = None
    camera_b = None
    
    try:
        print("Connecting to thermal camera A (device 1, COM4)...")
        camera_a = BosonWithTelemetry(device=1, port="COM4")
        print("Camera A connected successfully.")
    except Exception as e:
        print(f"Failed to connect to thermal camera A: {e}")
        return None, None
    
    try:
        print("Connecting to thermal camera B (device 2, COM6)...")
        camera_b = BosonWithTelemetry(device=2, port="COM6")
        print("Camera B connected successfully.")
    except Exception as e:
        print(f"Failed to connect to thermal camera B: {e}")
        if camera_a:
            camera_a.close()
        return None, None
    
    # Configure both cameras
    print("Performing flat field correction (FFC) on both cameras...")
    if force_ffc_at_init:
        camera_a.camera.do_ffc()
        camera_b.camera.do_ffc()
        time.sleep(1)  # Allow FFC to complete
        print("Forced FFC performed at initialization.")
    
    if disable_auto_ffc:
        camera_a.camera.set_ffc_manual()
        camera_b.camera.set_ffc_manual()
    
    print("Both cameras connected and configured successfully.")
    return camera_a, camera_b


def record_dual_thermal_data(camera_a: BosonWithTelemetry, 
                           camera_b: BosonWithTelemetry, 
                           duration: int, downsample: int) -> bool:
    """
    Record synchronized thermal data from both cameras.
    
    Args:
        camera_a: First camera object
        camera_b: Second camera object
        duration: Recording duration in seconds (-1 for manual stop)
        downsample: Temporal downsample factor for frames
        
    Returns:
        True if recording completed successfully, False otherwise
    """
    print(f"\nPreparing to record from both cameras...")
    if duration > 0:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Until manually stopped")
    
    print("Press Enter to start recording, or 'q' to cancel...")
    user_input = input().strip().lower()
    if user_input == 'q':
        print("Recording cancelled by user.")
        return False
    
    try:
        camera_a.set_downsample_factor(downsample)
        camera_b.set_downsample_factor(downsample)
        start_time = time.time()
        camera_a.start_logging()
        camera_b.start_logging()
        
        target_duration = duration if duration > 0 else float('inf')
        
        if duration == -1:
            print("Recording started. Press any key to stop...")
        else:
            print(f"Recording started for {duration} seconds. Press any key to stop early...")
        
        # Create a matplotlib figure for key detection
        fig = plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.title('Recording... Press any key to stop')
        plt.show(block=False)
        
        # Recording loop with progress tracking
        if duration > 0:
            with tqdm(total=duration, desc="Recording", unit="s") as pbar:
                while time.time() - start_time < target_duration:
                    elapsed = time.time() - start_time
                    pbar.update(elapsed - pbar.n)
                    
                    # Check for user input to stop early
                    if plt.waitforbuttonpress(timeout=0.1):
                        print("\nRecording stopped early by user.")
                        break
                    time.sleep(0.1)
        else:
            # Manual stop mode
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
        plt.close('all')
        return True  # Still save partial data
        
    finally:
        camera_a.stop_logging()
        camera_b.stop_logging()


def save_dual_data(camera_a: BosonWithTelemetry, 
                  camera_b: BosonWithTelemetry, 
                  output_file: str, compress: bool = False) -> None:
    """
    Save recorded dual camera data to file.
    
    Args:
        camera_a: First camera object with recorded data
        camera_b: Second camera object with recorded data
        output_file: Output file path
        compress: Whether to compress the data using zstandard
    """
    print("Processing and saving dual camera data...")
    
    # Convert data to numpy arrays
    raw_frames_a = np.array(camera_a.logged_images)
    timestamps_a = np.array(camera_a.logged_tstamps)
    timestamp_offset_a = camera_a.timestamp_offset
    
    raw_frames_b = np.array(camera_b.logged_images)
    timestamps_b = np.array(camera_b.logged_tstamps)
    timestamp_offset_b = camera_b.timestamp_offset
    
    print(f"Camera A captured {len(raw_frames_a)} frames")
    print(f"Camera B captured {len(raw_frames_b)} frames")
    
    np.savez(output_file,
                raw_thr_frames_A=raw_frames_a,
                raw_thr_tstamps_A=timestamps_a,
                thr_cam_timestamp_offset_A=timestamp_offset_a,
                raw_thr_frames_B=raw_frames_b,
                raw_thr_tstamps_B=timestamps_b,
                thr_cam_timestamp_offset_B=timestamp_offset_b)
    print(f"Dual camera data saved to: {output_file}")

    if compress:
        # Use command line to compress the file
        import subprocess
        compressed_file = output_file + ".zst"
        print(f"Compressing with zstd...")
        try:
            subprocess.run(["zstd", "-f", output_file, "-o", compressed_file], check=True)
            print(f"Compressed file saved to: {compressed_file}")
            output_file = compressed_file
            # Remove uncompressed file
            Path(output_file[:-4]).unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Compression failed: {e}")
        except FileNotFoundError:
            print("Warning: zstd command not found. Install zstd to enable compression.")
    
    # Print file size info
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    # Print synchronization info
    if len(timestamps_a) > 0 and len(timestamps_b) > 0:
        time_diff = abs(timestamps_a[0] - timestamps_b[0])
        print(f"Camera synchronization offset: {time_diff*1000:.1f} ms")


def cleanup_dual_cameras(camera_a: Optional[BosonWithTelemetry], 
                        camera_b: Optional[BosonWithTelemetry], leave_ffc_disabled: bool = False) -> None:
    """
    Properly cleanup both camera resources.
    
    Args:
        camera_a: First camera object to cleanup
        camera_b: Second camera object to cleanup
        leave_ffc_disabled: Whether to leave FFC disabled after recording
    """
    for camera, name in [(camera_a, "A"), (camera_b, "B")]:
        if camera:
            try:
                if not leave_ffc_disabled:
                    camera.camera.set_ffc_auto()
                    print(f"Camera {name} FFC reset to automatic mode.")
                camera.stop()
                camera.close()
                print(f"Camera {name} resources cleaned up.")
            except Exception as e:
                print(f"Warning: Error cleaning up camera {name}: {e}")


def main() -> None:
    """Main function to orchestrate the dual camera recording process."""
    args = parse_args()

    if args.downsample < 1:
        print("Invalid downsample factor. Must be >= 1.")
        sys.exit(1)

    # Prepare output path
    output_path = Path("./dual_data") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    camera_a = None
    camera_b = None
    
    try:
        # Initialize both cameras
        camera_a, camera_b = initialize_dual_cameras(args.disable_auto_ffc, args.force_ffc_at_init)
        if not camera_a or not camera_b:
            print("Failed to initialize both cameras. Exiting.")
            sys.exit(1)

        # Record data
        if record_dual_thermal_data(camera_a, camera_b, args.duration, args.downsample):
            save_dual_data(camera_a, camera_b, output_file, args.compress)
            print("Dual camera recording completed successfully!")
        else:
            print("Recording was cancelled.")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        cleanup_dual_cameras(camera_a, camera_b, args.leave_ffc_disabled)
        plt.close('all')  # Ensure all matplotlib figures are closed


if __name__ == "__main__":
    main()