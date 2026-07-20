# Thermal Camera Driver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A thin wrapper around [flirpy](https://github.com/LJMUAstroecology/flirpy)'s `ThreadedBoson` class for recording and previewing thermal video from FLIR Boson cameras, with telemetry (frame counter, timestamp) extraction. Main purpose is to record 60fps data from one or more cameras with minimal frame drop. 

## Installation

```bash
pip install -r requirements.txt
```

FLIR Boson camera(s) must be connected via USB and recognized by the OS (drivers installed separately if needed).

## Usage

### Live view

```bash
python wrapper_boson.py
```

Opens a window showing the live thermal feed in grayscale. Close the window (or Ctrl+C) to quit.

### Recording

Single camera, auto-detected:

```bash
python record_thermal_video.py --output recording.npz --duration 30
```

Multiple cameras, explicit device index and serial port (`DEVICE` or `DEVICE:PORT`, repeatable):

```bash
python record_thermal_video.py --camera 1:COM4 --camera 2:COM6 \
    --output dual_recording.npz --duration 60 --compress
```

Key flags:
- `--duration`: seconds to record, or `-1` to stop manually (default: 10)
- `--downsample`: temporal downsample factor (default: 1)
- `--compress`: compress the output with zstandard
- `--disable-auto-ffc` / `--force-ffc-at-init` / `--leave-ffc-disabled`: flat-field correction control

During recording, a small window pops up — press any key in it to stop early, or Ctrl+C.

### Previewing a recording

```bash
python preview_dataset.py path/to/recording.npz.zst
```

Browse frames with the slider or Left/Right arrow keys; see `python preview_dataset.py --help` for options (e.g. `--subtract-first-frame`).

## Data format

Recordings are saved as NumPy `.npz` archives (optionally zstandard-compressed to `.npz.zst`):

- Single camera: `raw_thr_frames`, `raw_thr_tstamps`, `thr_cam_timestamp_offset`, `dropped_frame_count`
- Multiple cameras: the same keys suffixed with `_0`, `_1`, ... per camera, in `--camera` order

Each frame is 640x514 (16-bit), with the first 2 rows holding telemetry data (frame counter, camera timestamp) and the remaining 512 rows holding the thermal image. `dropped_frame_count` is the number of camera frames missed during logging, detected via gaps in the telemetry frame counter.

## License

MIT — see [LICENSE](LICENSE). This applies to the code in this repository only.

This project depends on [flirpy](https://github.com/LJMUAstroecology/flirpy) (installed via `requirements.txt`, not bundled), which is separately MIT-licensed by Josh Veitch-Michaelis — see [flirpy's LICENSE.md](https://github.com/LJMUAstroecology/flirpy/blob/main/LICENSE.md) for its terms.
