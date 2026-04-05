"""
Data Analysis Example Script

This script demonstrates how to load and analyze thermal camera data
recorded with the thermal camera driver system.

Usage:
    python analyze_data.py --input recording.npz --show-stats

Author: [Your Name]
Date: January 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import zstandard as zstd
import zipfile


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze recorded thermal camera data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single camera data (compressed or uncompressed)
    python analyze_data.py --input recording.npz --show-stats
    
    # Analyze dual camera data
    python analyze_data.py --input dual_data/experiment.npz --dual --show-video
    
    # Export frames from compressed data
    python analyze_data.py --input compressed_recording.npz --export-frames --output-dir frames/
        """
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input .npz file path')
    parser.add_argument('--dual', action='store_true',
                       help='Data is from dual camera recording')
    parser.add_argument('--show-stats', action='store_true',
                       help='Display recording statistics')
    parser.add_argument('--show-video', action='store_true',
                       help='Play thermal video')
    parser.add_argument('--export-frames', action='store_true',
                       help='Export frames as PNG images')
    parser.add_argument('--output-dir', type=str, default='exported_frames',
                       help='Output directory for exported frames')
    return parser.parse_args()


def load_thermal_data(file_path: str, is_dual: bool = False) -> Dict[str, Any]:
    """
    Load thermal camera data from .npz file (compressed or uncompressed).
    
    Args:
        file_path: Path to the .npz data file
        is_dual: Whether data is from dual camera setup
        
    Returns:
        Dictionary containing loaded data
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    
    # Try to detect if file is compressed by attempting to decompress
    try:
        # First, try loading as regular npz file
        data = np.load(file_path)
        print("Loaded uncompressed data file")
        for key in data.files:
            print(f"Key: {key}")
            print(f"Shape: {data[key].shape}")
            print(f"Data type: {data[key].dtype}")
    except (ValueError, OSError, zipfile.BadZipFile) as e:
        # If that fails, try decompressing with zstandard first
        try:
            print("Attempting to decompress data file...")
            dctx = zstd.ZstdDecompressor()
            
            with open(file_path, 'rb') as compressed_file:
                with dctx.stream_reader(compressed_file) as reader:
                    decompressed_data = reader.read()
            
            # Load decompressed data using BytesIO
            import io
            with np.load(io.BytesIO(decompressed_data), allow_pickle=False) as npz_data:
                # Convert the NpzFile to a regular dictionary
                data = {key: npz_data[key] for key in npz_data.files}
            print("Successfully loaded compressed data file")
            
        except Exception as decomp_error:
            raise ValueError(f"Failed to load data file. Not a valid .npz file (compressed or uncompressed). "
                           f"Original error: {e}, Decompression error: {decomp_error}")
    
    if is_dual:
        required_keys = ['raw_thr_frames_A', 'raw_thr_tstamps_A', 'thr_cam_timestamp_offset_A',
                        'raw_thr_frames_B', 'raw_thr_tstamps_B', 'thr_cam_timestamp_offset_B']
    else:
        required_keys = ['raw_thr_frames', 'raw_thr_tstamps', 'thr_cam_timestamp_offset']
    
    # Check if all required keys exist
    data_keys = list(data.files) if hasattr(data, 'files') else list(data.keys())
    missing_keys = [key for key in required_keys if key not in data_keys]
    if missing_keys:
        raise ValueError(f"Missing keys in data file: {missing_keys}. Available keys: {data_keys}")
    
    return dict(data) if hasattr(data, 'files') else data


def display_statistics(data: Dict[str, Any], is_dual: bool = False) -> None:
    """
    Display recording statistics.
    
    Args:
        data: Loaded thermal data
        is_dual: Whether data is from dual camera setup
    """
    print("\n" + "="*50)
    print("RECORDING STATISTICS")
    print("="*50)
    
    if is_dual:
        frames_a = data['raw_thr_frames_A']
        timestamps_a = data['raw_thr_tstamps_A']
        frames_b = data['raw_thr_frames_B']  
        timestamps_b = data['raw_thr_tstamps_B']
        
        print(f"Camera A:")
        print(f"  Frames captured: {len(frames_a)}")
        print(f"  Frame shape: {frames_a[0].shape}")
        print(f"  Duration: {timestamps_a[-1] - timestamps_a[0]:.2f} seconds")
        print(f"  Average FPS: {len(frames_a) / (timestamps_a[-1] - timestamps_a[0]):.1f}")
        
        print(f"\nCamera B:")
        print(f"  Frames captured: {len(frames_b)}")
        print(f"  Frame shape: {frames_b[0].shape}")
        print(f"  Duration: {timestamps_b[-1] - timestamps_b[0]:.2f} seconds")
        print(f"  Average FPS: {len(frames_b) / (timestamps_b[-1] - timestamps_b[0]):.1f}")
        
        # Synchronization analysis
        sync_offset = abs(timestamps_a[0] - timestamps_b[0])
        print(f"\nSynchronization:")
        print(f"  Initial offset: {sync_offset*1000:.1f} ms")
        
        # Temperature statistics
        temp_stats_a = analyze_temperature_stats(frames_a)
        temp_stats_b = analyze_temperature_stats(frames_b)
        print(f"\nTemperature Statistics (Camera A):")
        print_temperature_stats(temp_stats_a)
        print(f"\nTemperature Statistics (Camera B):")
        print_temperature_stats(temp_stats_b)
        
    else:
        frames = data['raw_thr_frames']
        timestamps = data['raw_thr_tstamps']
        
        print(f"Frames captured: {len(frames)}")
        print(f"Frame shape: {frames[0].shape}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Average FPS: {len(frames) / (timestamps[-1] - timestamps[0]):.1f}")
        
        # Temperature statistics
        temp_stats = analyze_temperature_stats(frames)
        print(f"\nTemperature Statistics:")
        print_temperature_stats(temp_stats)


def strip_telemetry(frame: np.ndarray) -> np.ndarray:
    """
    Strip telemetry rows from a single thermal frame.

    Boson: telemetry is the first 2 rows  (height > 512, e.g. 514).
    Lepton: telemetry is the last 2 rows  (height <= 512, e.g. 122).
    """
    h = frame.shape[0]
    if h > 512:
        return frame[2:, ...]
    if h > 2:
        return frame[:-2, ...]
    return frame


def analyze_temperature_stats(frames: np.ndarray) -> Dict[str, float]:
    """
    Analyze temperature statistics from thermal frames.
    
    Args:
        frames: Array of thermal frames
        
    Returns:
        Dictionary with temperature statistics
    """
    thermal_frames = np.stack([strip_telemetry(f) for f in frames])
    
    # Calculate statistics
    min_temp = np.min(thermal_frames)
    max_temp = np.max(thermal_frames)
    mean_temp = np.mean(thermal_frames)
    std_temp = np.std(thermal_frames)
    
    return {
        'min': min_temp,
        'max': max_temp,
        'mean': mean_temp,
        'std': std_temp,
        'range': max_temp - min_temp
    }


def print_temperature_stats(stats: Dict[str, float]) -> None:
    """Print temperature statistics in a formatted way."""
    print(f"  Min value: {stats['min']:.0f}")
    print(f"  Max value: {stats['max']:.0f}")
    print(f"  Mean value: {stats['mean']:.1f}")
    print(f"  Std deviation: {stats['std']:.1f}")
    print(f"  Temperature range: {stats['range']:.0f}")


def play_thermal_video(data: Dict[str, Any], is_dual: bool = False) -> None:
    """
    Play thermal video with OpenCV.
    
    Args:
        data: Loaded thermal data
        is_dual: Whether data is from dual camera setup
    """
    print("\nPlaying thermal video. Press 'q' to quit, SPACE to pause.")
    
    if is_dual:
        frames_a = data['raw_thr_frames_A']
        frames_b = data['raw_thr_frames_B']
        
        cv2.namedWindow("Camera A", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Camera B", cv2.WINDOW_NORMAL)
        
        for i in range(min(len(frames_a), len(frames_b))):
            frame_a = normalize_frame(strip_telemetry(frames_a[i]))
            frame_b = normalize_frame(strip_telemetry(frames_b[i]))
            
            cv2.imshow("Camera A", frame_a)
            cv2.imshow("Camera B", frame_b)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause on space
                cv2.waitKey(0)
    else:
        frames = data['raw_thr_frames']
        cv2.namedWindow("Thermal Camera", cv2.WINDOW_NORMAL)
        
        for frame in frames:
            display_frame = normalize_frame(strip_telemetry(frame))
            cv2.imshow("Thermal Camera", display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause on space
                cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize thermal frame for display.
    
    Args:
        frame: Raw thermal frame (2-D single-channel expected)
        
    Returns:
        Normalized frame with colormap applied
    """
    if frame.ndim > 2:
        frame = frame.squeeze()
    if frame.ndim == 3:
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            frame = frame[:, :, 0]

    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_norm = np.uint8(frame_norm)
    frame_color = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
    
    return frame_color


def export_frames(data: Dict[str, Any], output_dir: str, is_dual: bool = False) -> None:
    """
    Export frames as PNG images.
    
    Args:
        data: Loaded thermal data
        output_dir: Output directory for frames
        is_dual: Whether data is from dual camera setup
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting frames to: {output_path}")
    
    if is_dual:
        frames_a = data['raw_thr_frames_A']
        frames_b = data['raw_thr_frames_B']
        
        for i in range(min(len(frames_a), len(frames_b))):
            frame_a = normalize_frame(strip_telemetry(frames_a[i]))
            cv2.imwrite(str(output_path / f"camera_A_frame_{i:06d}.png"), frame_a)
            
            frame_b = normalize_frame(strip_telemetry(frames_b[i]))
            cv2.imwrite(str(output_path / f"camera_B_frame_{i:06d}.png"), frame_b)
    else:
        frames = data['raw_thr_frames']
        
        for i, frame in enumerate(frames):
            frame_color = normalize_frame(strip_telemetry(frame))
            cv2.imwrite(str(output_path / f"frame_{i:06d}.png"), frame_color)
    
    print(f"Exported {len(frames) if not is_dual else min(len(frames_a), len(frames_b))} frames")


def main() -> None:
    """Main function."""
    args = parse_args()
    
    try:
        # Load data
        data = load_thermal_data(args.input, args.dual)
        print("Data loaded successfully!")
        
        # Display statistics
        if args.show_stats:
            display_statistics(data, args.dual)
        
        # Play video
        if args.show_video:
            play_thermal_video(data, args.dual)
        
        # Export frames
        if args.export_frames:
            export_frames(data, args.output_dir, args.dual)
            
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()