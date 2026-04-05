"""
FLIR Boson Thermal Camera Wrapper with Telemetry Support

This module provides a wrapper class for FLIR Boson thermal cameras that extends
the base ThreadedBoson class with telemetry logging capabilities.

Author: [Your Name]
Date: January 2026
"""

import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple

from flirpy.camera.threadedboson import ThreadedBoson


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
        self.camera.do_ffc()  # Perform initial flat field correction
        
        # Initialize logging attributes
        self.logged_images = []
        self.logged_tstamps = []
        self.logged_cam_tstamps = []
        self.logged_frame_numbers = []
        self.enable_logging = False

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            self.stop()
        except Exception:
            pass
        if getattr(self, "camera", None) is not None:
            try:
                self.camera.close()
            except Exception:
                pass
    
    def stop_logging(self) -> None:
        """Stop logging thermal frames and timestamps."""
        self.enable_logging = False

    def start_logging(self) -> None:
        """Start logging thermal frames and timestamps."""
        self.enable_logging = True
        self.add_post_callback(self.post_cap_hook)
    
    def post_cap_hook(self, image: np.ndarray) -> None:
        """
        Callback function executed after each frame capture.

        Logs the full frame (with telemetry header rows) and parses
        camera-internal timestamps and frame numbers from the telemetry,
        mirroring LeptonWrapper's _post_capture_hook behaviour.

        Args:
            image: Captured thermal image array (includes 2 telemetry rows)
        """
        if self.enable_logging:
            capture_time = time.time()
            self.logged_images.append(image)
            self.logged_tstamps.append(capture_time)

            try:
                telemetry = image[:2, :, 0] if image.ndim == 3 else image[:2, :]
                frame_number, cam_timestamp = self.parse_telemetry(telemetry)
                self.logged_cam_tstamps.append(cam_timestamp + self.timestamp_offset)
                self.logged_frame_numbers.append(frame_number)
            except Exception:
                self.logged_cam_tstamps.append(capture_time)
                self.logged_frame_numbers.append(-1)

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
        frame_counter = telemetry[0, 42] * 2**16 + telemetry[0, 43]
        timestamp_in_ms = telemetry[0, 140] * 2**16 + telemetry[0, 141]
        timestamp = timestamp_in_ms / 1000.0
        return frame_counter, timestamp

    

if __name__ == "__main__":
    """
    Demo script for live thermal camera visualization.
    
    Displays thermal camera feed with turbo colormap in real-time.
    Press 'q' to quit the application.
    """
    print("Starting thermal camera live view...")
    print("Press 'q' to quit")
    
    try:
        boson = BosonWithTelemetry()
        cv2.namedWindow("Boson Thermal Camera", cv2.WINDOW_NORMAL)
        
        while True:
            image, timestamp, frame_number, _ = boson.get_next_image()

            # Normalize and convert to 8-bit for display
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = np.uint8(image)
            image = cv2.flip(image, 1)  # Mirror image for natural viewing
            image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
            
            # Print frame info (overwrite previous line)
            print(f"Timestamp: {timestamp:.3f}, Frame: {frame_number}\r", end="")

            cv2.imshow("Boson Thermal Camera", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the thermal camera is connected and accessible.")
    
    finally:
        if 'boson' in locals():
            boson.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")