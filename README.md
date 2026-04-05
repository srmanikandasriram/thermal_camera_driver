# Thermal Camera Driver

A Python driver system for FLIR thermal cameras. Supports **FLIR Boson** (with telemetry) and **FLIR Lepton** cameras for recording thermal video data.

## Features

- Support for single and dual FLIR Boson camera recording
- Support for FLIR Lepton camera recording
- Simultaneous Boson + Lepton recording with synchronized telemetry
- Real-time thermal video display with colormap visualization
- Telemetry data extraction for Boson (timestamps, frame numbers)
- Data compression support using Zstandard
- Configurable recording duration
- Manual FFC (Flat Field Correction) control for Boson

## Hardware Requirements

- FLIR Boson or FLIR Lepton thermal camera
- USB connection (Lepton also supports SPI)
- Compatible operating system (Windows, Linux, macOS)

## Installation

### 1. Create a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv thermal_camera_env

# Activate virtual environment
# On Windows:
thermal_camera_env\Scripts\activate
# On macOS/Linux:
source thermal_camera_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Hardware Setup

**Boson:** Connect via USB and ensure FLIR drivers are installed.

**Lepton:** Connect via USB (PureThermal board) or SPI. No additional drivers are required beyond `flirpy`.

## Usage

### Single Camera Recording

Record thermal video from a single Boson camera:

```bash
python record_thermal_video.py --output "recordings/thermal_video.npz" --duration 30
```

**Parameters:**
- `--output`: Output file path (required)
- `--duration`: Recording duration in seconds (default: 10)
- `--compress`: Enable zstandard compression (optional)

**Example:**
```bash
# Record 60 seconds with compression
python record_thermal_video.py --output "data/test_recording.npz" --duration 60 --compress
```

### Lepton Camera Recording

Record thermal video from a FLIR Lepton camera:

```bash
python record_thermal_video_lepton.py --output "lepton_recordings/recording.npz" --duration 30
```

**Parameters:**
- `--output`: Output file path (required)
- `--duration`: Recording duration in seconds (default: 10)
- `--compress`: Enable zstandard compression (optional)

**Example:**
```bash
# Record 60 seconds with compression
python record_thermal_video_lepton.py --output "lepton_recordings/test.npz" --duration 60 --compress
```

The Lepton handles FFC (flat field correction) automatically — no manual calibration step is needed.

### Live Preview (Lepton)

```bash
python wrapper_lepton.py
```

Displays the live feed with an inferno colormap. Press 'q' to quit.

### Dual Camera Recording (Boson)

Record synchronized thermal video from two Boson cameras:

```bash
python record_dual_thermal_video.py --output "dual_recording.npz" --duration 120
```

**Parameters:**
- `--output`: Output file name (required)
- `--duration`: Recording duration in seconds (default: -1 for manual stop)

**Example:**
```bash
# Record until manually stopped
python record_dual_thermal_video.py --output "experiment_001.npz"
```

### Boson + Lepton Simultaneous Recording

Record synchronized thermal video from a Boson and a Lepton camera:

```bash
python record_boson_lepton_video.py --output "recording.npz" --duration 60
```

**Parameters:**
- `--output`: Output file name (required, stored in `./boson_lepton_data/`)
- `--duration`: Recording duration in seconds (default: -1 for manual stop)
- `--compress`: Enable zstandard compression (optional)

**Example:**
```bash
# Record 2 minutes with compression
python record_boson_lepton_video.py --output "experiment_001.npz" --duration 120 --compress

# Record until manually stopped
python record_boson_lepton_video.py --output "continuous.npz"
```

The Boson is expected on device 1 / COM4 by default (edit `initialize_cameras()` if your setup differs). The Lepton is auto-detected.

### Live Preview (Boson)

```bash
python wrapper_boson.py
```

Displays the live feed with a turbo colormap. Press 'q' to quit.

## Data Format

Recorded data is saved as NumPy archive files (.npz) containing:

### Single Camera:
- `raw_thr_frames`: Array of thermal frames
- `raw_thr_tstamps`: Timestamps for each frame
- `thr_cam_timestamp_offset`: Camera timestamp offset

### Dual Boson Camera:
- `raw_thr_frames_A/B`: Arrays of thermal frames from camera A/B
- `raw_thr_tstamps_A/B`: Timestamps for each frame from camera A/B
- `thr_cam_timestamp_offset_A/B`: Camera timestamp offsets

### Boson + Lepton:
- `raw_thr_frames_boson`: Boson frames (with 2-row telemetry header)
- `raw_thr_tstamps_boson`: Boson system timestamps
- `thr_cam_timestamp_offset_boson`: Boson system−camera time offset
- `raw_thr_frames_lepton`: Lepton frames (with 2-row telemetry footer)
- `raw_thr_tstamps_lepton`: Lepton system timestamps
- `thr_cam_timestamp_offset_lepton`: Lepton system−camera time offset
- `raw_thr_cam_tstamps_lepton`: Lepton camera-clock timestamps
- `raw_thr_frame_numbers_lepton`: Lepton frame counter from telemetry

## Data Analysis

### Loading Recorded Data

```python
import numpy as np

# Load single camera data
data = np.load('recording.npz')
frames = data['raw_thr_frames']
timestamps = data['raw_thr_tstamps']
offset = data['thr_cam_timestamp_offset']

# Load dual Boson data
dual_data = np.load('dual_recording.npz')
frames_a = dual_data['raw_thr_frames_A']
frames_b = dual_data['raw_thr_frames_B']

# Load Boson + Lepton data
bl_data = np.load('boson_lepton_recording.npz')
boson_frames = bl_data['raw_thr_frames_boson']
lepton_frames = bl_data['raw_thr_frames_lepton']
lepton_cam_ts = bl_data['raw_thr_cam_tstamps_lepton']
```

### Analysis Tools

Use the included analysis script for comprehensive data analysis:

```bash
# Display recording statistics
python analyze_data.py --input recording.npz --show-stats

# Play thermal video
python analyze_data.py --input recording.npz --show-video

# Export frames as PNG images
python analyze_data.py --input recording.npz --export-frames --output-dir frames/

# Analyze dual camera data
python analyze_data.py --input dual_data/experiment.npz --dual --show-stats
```

## Project Structure

```
thermal_camera_driver/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup_environment.py          # Environment setup script
├── config.ini                    # Configuration settings
├── .gitignore                    # Git ignore rules
│
├── wrapper_boson.py                   # Boson camera wrapper class
├── wrapper_lepton.py                  # Lepton camera wrapper class
├── record_thermal_video.py            # Boson single camera recording script
├── record_thermal_video_lepton.py     # Lepton recording script
├── record_dual_thermal_video.py       # Boson dual camera recording script
├── record_boson_lepton_video.py       # Boson + Lepton simultaneous recording
├── analyze_data.py                    # Data analysis and visualization
├── test_camera.py                     # Camera connection test script
│
├── recordings/                        # Boson recordings (auto-created)
├── lepton_recordings/                 # Lepton recordings (auto-created)
├── dual_data/                         # Dual Boson recordings (auto-created)
├── boson_lepton_data/                 # Boson + Lepton recordings (auto-created)
└── exported_frames/                   # Exported frame images (auto-created)
```

## Quick Start

1. **Automated Setup** (Recommended):
```bash
# Clone or download the project
# Navigate to the project directory
python setup_environment.py
```

2. **Manual Setup**:
```bash
# Create virtual environment
python -m venv thermal_camera_env

# Activate virtual environment
# Windows:
thermal_camera_env\Scripts\activate
# macOS/Linux:
source thermal_camera_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Test Camera Connection**:
```bash
# Test single camera
python test_camera.py

# Test dual camera setup
python test_camera.py --dual
```

4. **Start Recording**:
```bash
# Boson single camera
python record_thermal_video.py --output my_recording.npz --duration 30

# Lepton camera
python record_thermal_video_lepton.py --output lepton_recording.npz --duration 30

# Boson dual camera
python record_dual_thermal_video.py --output dual_recording.npz --duration 60

# Boson + Lepton simultaneous
python record_boson_lepton_video.py --output mixed_recording.npz --duration 60
```

## Configuration

### Boson Camera Settings

The system automatically configures the Boson with these settings:
- Resolution: 640x514 pixels (including 2-row telemetry)
- Format: Y16 (16-bit grayscale)
- Buffer size: 1 frame
- FFC: Manual mode (performed at startup)

### Lepton Camera Settings

The Lepton is configured automatically by `flirpy`:
- Resolution: 160x120 (Lepton 3/3.5) or 80x60 (Lepton 2)
- FFC: Automatic (handled by the camera hardware)
- Timestamps: System clock (`time.time()`)

### Telemetry Data

**Boson:** Telemetry is embedded in the first 2 rows of each frame:
- Frame counter (words 42-43)
- Timestamp in milliseconds (words 140-141)

**Lepton:** Telemetry is embedded in the last 2 rows (footer) of each frame and parsed automatically by `LeptonWrapper`. Extracted fields include uptime, frame count, FPA/housing temperatures, FFC status, and AGC state. Camera-clock timestamps and frame numbers are saved as separate arrays in the .npz output.

## Troubleshooting

### Camera Connection Issues

**Boson:**
- **Run the test script first**: `python test_camera.py`
- Verify the camera is properly connected via USB
- Check that FLIR drivers are installed
- Ensure no other applications are using the camera
- Try different USB ports or cables
- For dual cameras, verify COM port assignments in [config.ini](config.ini)

**Lepton:**
- Ensure `flirpy` is installed: `pip install flirpy`
- Verify the PureThermal board (or equivalent) is connected via USB
- On Linux, you may need to add your user to the `video` group: `sudo usermod -aG video $USER`
- Note: the Lepton performs automatic FFC (shutter events) periodically — brief frame artifacts during FFC are normal

### Permission Errors
- Run the application as administrator (Windows) or with sudo (Linux)
- Check camera device permissions

### Dependencies Issues
- Use the automated setup: `python setup_environment.py`
- Manually verify packages: `pip list`
- Ensure Python version 3.7+ is being used

### Recording Issues
- Check available disk space for large recordings
- Ensure output directory exists and is writable
- For dual camera setup, verify both cameras are connected and accessible
- Monitor system resources during long recordings

### Performance Issues
- Close unnecessary applications to free system resources
- Use compression for long recordings: `--compress`
- Consider reducing recording resolution if needed
- Monitor CPU and memory usage during recording

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate documentation
5. Submit a pull request

### Code Style

- Follow PEP 8 Python style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write descriptive variable names

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add contact information or support channels here]