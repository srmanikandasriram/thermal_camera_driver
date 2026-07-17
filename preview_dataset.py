#!/usr/bin/env python3
"""Preview a raw FLIR .npz.zst thermal recording.

Handles both single recordings (keys `raw_thr_frames`, `raw_thr_tstamps`)
and dual recordings (same keys with `_A`/`_B` suffixes). The telemetry
header (first two rows of each frame) is stripped before display.
`raw_thr_tstamps` is already an absolute software timestamp (the
`thr_cam_timestamp_offset` field is only for cross-checking against the
camera's telemetry and is not applied here); it is shown alongside the
frame number as human-readable local time.

Usage:
    python preview_dataset.py path/to/scene.npz.zst
    python preview_dataset.py path/to/scene.npz.zst --frame 100
    python preview_dataset.py path/to/scene.npz.zst --subtract-first-frame

Navigate with the Left/Right arrow keys or the slider once the window is open.
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd
from matplotlib.widgets import Slider

TELEMETRY_ROWS = 2


def load_npz(path: Path) -> dict:
    with open(path, "rb") as fh:
        raw = zstd.ZstdDecompressor().decompress(fh.read())
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def find_channels(data: dict) -> list[str]:
    if "raw_thr_frames" in data:
        return [""]
    suffixes = sorted(
        k[len("raw_thr_frames"):] for k in data if k.startswith("raw_thr_frames_")
    )
    if not suffixes:
        raise ValueError(f"No raw_thr_frames key found. Available keys: {sorted(data)}")
    return suffixes


def strip_telemetry(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    return frame[TELEMETRY_ROWS:, :]


def format_time(tstamp: float) -> str:
    return datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to a .npz.zst recording")
    parser.add_argument("--frame", type=int, default=None, help="Initial frame index (default: middle frame)")
    parser.add_argument(
        "--subtract-first-frame",
        action="store_true",
        help="Subtract each channel's first frame before displaying",
    )
    args = parser.parse_args()

    data = load_npz(args.npz_path)
    channels = find_channels(data)

    frames = {c: data[f"raw_thr_frames{c}"] for c in channels}
    tstamps = {c: data[f"raw_thr_tstamps{c}"] for c in channels}
    first_frames = {c: strip_telemetry(frames[c][0]).astype(np.int32) for c in channels}

    def get_frame(c: str, idx: int) -> np.ndarray:
        img = strip_telemetry(frames[c][idx]).astype(np.int32)
        if args.subtract_first_frame:
            img = img - first_frames[c]
        return img

    n_frames = min(frames[c].shape[0] for c in channels)
    state = {"idx": args.frame if args.frame is not None else n_frames // 2}
    state["idx"] = max(0, min(n_frames - 1, state["idx"]))

    fig, axes = plt.subplots(1, len(channels), squeeze=False, figsize=(6 * len(channels), 6.5))
    axes = axes[0]
    fig.subplots_adjust(bottom=0.15)

    cmap = "gray" if not args.subtract_first_frame else "RdBu_r"
    images = []
    for ax, c in zip(axes, channels):
        img = get_frame(c, state["idx"])
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(c.lstrip("_") or "recording")
        images.append(im)

    slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.04])
    slider = Slider(slider_ax, "Frame", 0, n_frames - 1, valinit=state["idx"], valstep=1)

    _updating = {"flag": False}

    def redraw():
        idx = state["idx"]
        for im, ax, c in zip(images, axes, channels):
            img = get_frame(c, idx)
            im.set_data(img)
            if args.subtract_first_frame:
                bound = max(abs(img.min()), abs(img.max()), 1)
                im.set_clim(-bound, bound)
            else:
                im.set_clim(img.min(), img.max())
        titles = []
        for c in channels:
            label = c.lstrip("_") or "recording"
            t_str = format_time(tstamps[c][idx])
            titles.append(f"{label}: frame {idx}/{n_frames - 1}  {t_str}")
        fig.suptitle("   |   ".join(titles))
        _updating["flag"] = True
        slider.set_val(idx)
        _updating["flag"] = False
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["idx"] = min(n_frames - 1, state["idx"] + 1)
        elif event.key == "left":
            state["idx"] = max(0, state["idx"] - 1)
        else:
            return
        redraw()

    def on_slider(val):
        if _updating["flag"]:
            return
        state["idx"] = int(val)
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    slider.on_changed(on_slider)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
