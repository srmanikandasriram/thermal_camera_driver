"""
Single Thermal Camera Recording Script

This script records thermal video data from a single FLIR Boson camera
with optional compression and configurable duration.

Usage:
    python record_thermal_video.py --output recording.npz --duration 30 --compress

Author: [Your Name]
Date: January 2026
"""

import sys
import argparse
import time
from typing import Optional
import cv2
import numpy as np
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
        description="Record thermal video from a FLIR Boson camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record 30 seconds to file
    python record_thermal_video.py --output thermal_data.npz --duration 30
    
    # Record with compression
    python record_thermal_video.py --output compressed_data.npz --duration 60 --compress
    
    # Record to specific directory
    python record_thermal_video.py --output recordings/experiment_001.npz --duration 120
        """
    )
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (will create directories if needed)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration of recording in seconds (default: 10)')
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


def initialize_camera(disable_auto_ffc: bool, force_ffc_at_init: bool) -> Optional[BosonWithTelemetry]:
    """
    Initialize and configure the thermal camera.
    
    Args:
        disable_auto_ffc: Whether to disable automatic FFC during recording
        force_ffc_at_init: Whether to force FFC at initialization 
        
    Returns:
        Configured camera object or None if initialization fails
    """
    try:
        print("Connecting to thermal camera...")
        camera = BosonWithTelemetry()
        
        if disable_auto_ffc:
            # Disable automatic FFC during recording
            camera.camera.set_ffc_manual()

        if force_ffc_at_init:
            # Force FFC at initialization
            camera.camera.do_ffc()
            time.sleep(1)  # Allow FFC to complete
            print("Forced FFC performed at initialization.")

        print("Camera connected successfully.")
        return camera
        
    except Exception as e:
        print(f"Failed to connect to thermal camera: {e}")
        print("Please check:")
        print("- Camera is properly connected via USB")
        print("- Camera drivers are installed")
        print("- No other applications are using the camera")
        return None


def record_thermal_data(camera: BosonWithTelemetry, duration: int, downsample: int) -> bool:
    """
    Record thermal data for the specified duration.
    
    Args:
        camera: Initialized camera object
        duration: Recording duration in seconds
        downsample: Temporal downsample factor for frames
        
    Returns:
        True if recording completed successfully, False otherwise
    """
    print(f"\nPreparing to record for {duration} seconds...")
    print("Press Enter to start recording, or 'q' to cancel...")
    
    user_input = input().strip().lower()
    if user_input == 'q':
        print("Recording cancelled by user.")
        return False
    
    camera.set_downsample_factor(downsample)
    start_time = time.time()
    camera.start_logging()
    print("Recording started. Press 'q' to stop early.")

    try:
        with tqdm(total=duration, desc="Recording", unit="s") as pbar:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                pbar.update(elapsed - pbar.n)
                
                # Check for early termination
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nRecording stopped early by user.")
                    break
                    
                time.sleep(0.1)
                
        return True
        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        return True  # Still save partial data
        
    finally:
        camera.stop_logging()


def save_data(camera: BosonWithTelemetry, output_file: str, compress: bool) -> None:
    """
    Save recorded thermal data to file.
    
    Args:
        camera: Camera object containing recorded data
        output_file: Output file path
        compress: Whether to compress the data
    """
    print("Processing and saving data...")
    
    # Convert lists to numpy arrays
    raw_frames = np.array(camera.logged_images)
    timestamps = np.array(camera.logged_tstamps)
    timestamp_offset = camera.timestamp_offset
    
    print(f"Captured {len(raw_frames)} frames")
    
    np.savez(output_file, 
            raw_thr_frames=raw_frames,
            raw_thr_tstamps=timestamps,
            thr_cam_timestamp_offset=timestamp_offset)
    print(f"Data saved to: {output_file}")

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


def cleanup_camera(camera: Optional[BosonWithTelemetry], leave_ffc_disabled: bool = False) -> None:
    """
    Properly cleanup camera resources.
    
    Args:
        camera: Camera object to cleanup (can be None)
        leave_ffc_disabled: Whether to leave FFC disabled after recording
    """
    if camera:
        try:
            if not leave_ffc_disabled:
                camera.camera.set_ffc_auto()
                print("FFC reset to automatic mode.")
            camera.stop()
            camera.close()
            print("Camera resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error during camera cleanup: {e}")


def main() -> None:
    """Main function to orchestrate the recording process."""
    args = parse_args()

    if args.downsample < 1:
        print("Error: Downsample factor must be >= 1")
        sys.exit(1)

    # Prepare output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    camera = None
    try:
        # Initialize camera
        camera = initialize_camera(args.enable_auto_ffc, args.force_ffc_at_init)
        if not camera:
            sys.exit(1)

        # Record data
        if record_thermal_data(camera, args.duration, args.downsample):
            save_data(camera, output_file, args.compress)
            print("Recording completed successfully!")
        else:
            print("Recording was cancelled.")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        cleanup_camera(camera, args.leave_ffc_disabled)


if __name__ == "__main__":
    main()
