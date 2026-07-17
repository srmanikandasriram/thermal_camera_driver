"""
FLIR Boson Thermal Camera Wrapper with Telemetry Support

This module provides a wrapper class for FLIR Boson thermal cameras that extends
the base ThreadedBoson class with telemetry logging capabilities.
"""

import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple

from flirpy.camera.threadedboson import ThreadedBoson

logger = logging.getLogger(__name__)


class BosonWithTelemetry(ThreadedBoson):
    """
    Enhanced FLIR Boson camera wrapper with telemetry logging support.
    
    This class extends ThreadedBoson to provide additional functionality for:
    - Logging thermal frames and timestamps
    - Extracting telemetry data from camera frames
    - Synchronized timestamp management
    
    Attributes:
        logged_images (list): List of captured thermal frames
        logged_tstamps (list): List of corresponding timestamps
        enable_logging (bool): Flag to control data logging
        timestamp_offset (float): Offset between system and camera timestamps
    """
    def __init__(self, device: Optional[int] = None, port: Optional[str] = None, 
                 baudrate: int = 921600, loglevel: int = logging.WARNING):
        """
        Initialize the Boson camera with telemetry support.
        
        Args:
            device: Camera device index (default: None for auto-detection)
            port: Serial port for communication (default: None)
            baudrate: Serial communication baud rate (default: 921600)
            loglevel: Logging level (default: logging.WARNING)
        """
        super().__init__(device=device, port=port, baudrate=baudrate, loglevel=loglevel)

        self.configure()
        self.start()

        # Initialize logging attributes
        self.logged_images = []
        self.logged_tstamps = []
        self.enable_logging = False
        self.downsample_factor = 1 # No downsampling by default

        # Dropped-frame tracking
        self.last_frame_counter: Optional[int] = None
        self.dropped_frame_count = 0

    def set_downsample_factor(self, factor: int) -> None:
        """
        Set the downsample factor for captured frames.
        
        Args:
            factor: Downsample factor (must be >= 1)
        """
        if factor < 1:
            raise ValueError("Downsample factor must be >= 1")
        self.downsample_factor = factor

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        self.stop()
        self.camera.close()
    
    def stop_logging(self) -> None:
        """Stop logging thermal frames and timestamps."""
        self.enable_logging = False

    def start_logging(self) -> None:
        """Start logging thermal frames and timestamps."""
        self.last_frame_counter = None
        self.dropped_frame_count = 0
        self.enable_logging = True
        self.add_post_callback(self.post_cap_hook)

    def post_cap_hook(self, image: np.ndarray) -> None:
        """
        Callback function executed after each frame capture.

        Args:
            image: Captured thermal image array
        """
        if self.enable_logging:
            frame_counter, _ = self.parse_telemetry(image[:2, :, 0])
            self._check_dropped_frames(frame_counter)

            if self.downsample_factor > 1 and frame_counter % self.downsample_factor != 0:
                return  # Skip this frame based on downsample factor
            self.logged_images.append(image)
            self.logged_tstamps.append(time.time())

    def _check_dropped_frames(self, frame_counter: int) -> None:
        """
        Compare the camera's frame counter to the previous one and track any gap.

        The camera's frame counter increments once per native camera frame
        regardless of temporal downsampling, so a gap here means the driver
        missed frames rather than that they were intentionally skipped.

        Args:
            frame_counter: Frame counter parsed from the current frame's telemetry
        """
        if self.last_frame_counter is not None:
            gap = frame_counter - self.last_frame_counter - 1
            if gap > 0:
                self.dropped_frame_count += gap
                logger.warning(
                    f"Dropped {gap} frame(s): counter jumped from "
                    f"{self.last_frame_counter} to {frame_counter}"
                )
        self.last_frame_counter = frame_counter

    def compute_timestamp_offset(self) -> None:
        """
        Compute the offset between system time and camera timestamp.
        This ensures synchronized timestamps across different time sources.
        """
        (_, latest_image), system_time = self.camera.cap.read(), time.time()
        _, cam_timestamp = self.parse_telemetry(latest_image[:2, :])
        self.timestamp_offset = system_time - cam_timestamp

    def configure(self) -> None:
        """
        Configure camera settings for optimal thermal imaging.
        
        Sets up:
        - Frame dimensions (640x514 including telemetry rows)
        - Y16 format for 16-bit thermal data
        - Buffer size and RGB conversion settings
        """
        self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 514)
        self.camera.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        self.camera.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.camera.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.cap.grab()

        self.compute_timestamp_offset()
    
    def get_next_image(self, hflip: bool = False) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Get the next thermal image with telemetry data.
        
        Args:
            hflip: Whether to horizontally flip the image (default: False)
            
        Returns:
            Tuple containing:
            - image: Thermal image array (512x640)
            - timestamp: Synchronized timestamp
            - frame_number: Camera frame counter
            - telemetry: Raw telemetry data array
        """
        latest_image = self.latest()

        telemetry = latest_image[:2, :, 0]
        image = latest_image[2:, :, 0]
        frame_number, cam_timestamp = self.parse_telemetry(telemetry)
        timestamp = cam_timestamp + self.timestamp_offset

        if hflip:
            image = cv2.flip(image, 1)
        return image, timestamp, frame_number, telemetry

    def parse_telemetry(self, telemetry: np.ndarray) -> Tuple[int, float]:
        """
        Parse telemetry data from the camera frame header.
        
        The telemetry data is embedded in the first two rows of each frame
        and contains frame counters and timestamps.
        
        Args:
            telemetry: Telemetry data array (2x640)
            
        Returns:
            Tuple containing:
            - frame_counter: Sequential frame number from camera
            - timestamp: Camera timestamp in seconds
        """
        frame_counter = int(telemetry[0, 42]) * 2**16 + int(telemetry[0, 43])
        timestamp_in_ms = int(telemetry[0, 140]) * 2**16 + int(telemetry[0, 141])
        timestamp = timestamp_in_ms / 1000.0
        return frame_counter, timestamp
    
    def _grab(self):
        image = np.expand_dims(self.camera.grab(), -1)
        return image

    

if __name__ == "__main__":
    """
    Demo script for live thermal camera visualization.

    Displays thermal camera feed with turbo colormap in real-time.
    Close the window (or Ctrl+C) to quit the application.
    """
    import matplotlib.pyplot as plt

    print("Starting thermal camera live view...")
    print("Close the window to quit")

    closed = {"flag": False}

    def on_close(event):
        closed["flag"] = True

    boson = None
    try:
        boson = BosonWithTelemetry()

        image, timestamp, frame_number, _ = boson.get_next_image(hflip=True)
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("close_event", on_close)
        im = ax.imshow(image, cmap="gray")
        title = ax.set_title(f"Timestamp: {timestamp:.3f}, Frame: {frame_number}")
        plt.show(block=False)

        while not closed["flag"]:
            image, timestamp, frame_number, _ = boson.get_next_image(hflip=True)
            im.set_data(image)
            im.set_clim(image.min(), image.max())
            title.set_text(f"Timestamp: {timestamp:.3f}, Frame: {frame_number}")
            fig.canvas.draw_idle()
            plt.pause(0.001)

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the thermal camera is connected and accessible.")

    finally:
        if boson is not None:
            boson.stop()
        plt.close('all')
        print("Camera stopped and windows closed.")