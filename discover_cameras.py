"""
Thermal Camera Discovery Script

Lists active serial (COM) ports and available video devices, flagging any
that match the FLIR Boson's USB vendor/product ID (0x09CB / 0x4007) so they
can be passed to record_thermal_video.py via --camera DEVICE:PORT.

Usage:
    python discover_cameras.py
"""

import sys
from dataclasses import dataclass
from typing import List, Optional

from serial.tools import list_ports

BOSON_VID = 0x09CB
BOSON_PID = 0x4007


@dataclass
class UsbLocation:
    physical_id: str  # USB device serial number; stable key for pairing interfaces
    root_hub: str      # e.g. "usb3" (the host controller's root hub)
    port_path: str      # e.g. "3-10.4" (bus-port[.port...], encodes the hub chain)


@dataclass
class SerialPortInfo:
    device: str
    description: str
    vid: Optional[int]
    pid: Optional[int]
    usb_location: Optional[UsbLocation] = None

    @property
    def is_boson(self) -> bool:
        return self.vid == BOSON_VID and self.pid == BOSON_PID

    @property
    def physical_id(self) -> Optional[str]:
        return self.usb_location.physical_id if self.usb_location else None


@dataclass
class VideoDeviceInfo:
    device: int
    name: str
    is_boson: bool
    usb_location: Optional[UsbLocation] = None

    @property
    def physical_id(self) -> Optional[str]:
        return self.usb_location.physical_id if self.usb_location else None


def _usb_location(sys_path: str) -> Optional[UsbLocation]:
    """
    Identify the physical USB device (composite device, not a single USB
    interface) backing a /sys/class/... node, e.g. a Boson exposes one video
    interface and one serial interface off the *same* USB device, and also
    report which root hub/port it's plugged into. Returns None if it can't be
    determined (e.g. non-Linux, non-USB).
    """
    try:
        import pyudev
    except ImportError:
        return None

    try:
        context = pyudev.Context()
        udev = pyudev.Devices.from_path(context, sys_path)
        usb_device = udev.find_parent("usb", "usb_device")
        if usb_device is None:
            return None

        serial = usb_device.properties.get("ID_SERIAL_SHORT")
        if not serial:
            return None

        root_hub = usb_device
        while True:
            parent = root_hub.find_parent("usb")
            if parent is None or parent.sys_name == root_hub.sys_name:
                break
            root_hub = parent

        return UsbLocation(physical_id=serial, root_hub=root_hub.sys_name,
                            port_path=usb_device.sys_name)
    except Exception:
        return None


def find_serial_ports() -> List[SerialPortInfo]:
    """List all active serial ports, regardless of whether they're a Boson."""
    ports = []
    for port in list_ports.comports():
        usb_location = _usb_location(f"/sys/class/tty/{port.name}") if sys.platform.startswith("linux") else None
        ports.append(SerialPortInfo(device=port.device, description=port.description or "",
                                     vid=port.vid, pid=port.pid, usb_location=usb_location))
    return ports


def find_video_devices() -> List[VideoDeviceInfo]:
    """
    List available video devices.

    Uses Video4Linux/udev on Linux (matching flirpy's own device detection),
    since that's the only platform that exposes USB VID/PID per video device
    without extra native tooling.
    """
    if sys.platform.startswith("linux"):
        return _find_video_devices_linux()
    else:
        print(f"Warning: video device enumeration is not implemented for "
              f"platform '{sys.platform}'; only serial ports will be listed.",
              file=sys.stderr)
        return []


def _find_video_devices_linux() -> List[VideoDeviceInfo]:
    import os
    import pyudev

    path = "/sys/class/video4linux/"
    if not os.path.exists(path):
        print("Warning: Video4Linux not found", file=sys.stderr)
        return []

    context = pyudev.Context()
    devices = []
    for name in sorted(os.listdir(path)):
        if not name.startswith("video"):
            continue
        udev = pyudev.Devices.from_path(context, os.path.join(path, name))
        try:
            device_num = int(name[len("video"):])
        except ValueError:
            continue

        vid = udev.properties.get("ID_VENDOR_ID")
        pid = udev.properties.get("ID_MODEL_ID")
        model = udev.properties.get("ID_MODEL", "Unknown")
        is_boson = bool(vid and pid and vid.lower() == "09cb" and pid.lower() == "4007")
        # Each Boson exposes multiple video nodes (only one of which actually
        # streams) *and* a serial port, all off the same physical USB device.
        # physical_id groups/pairs them (see _usb_location).
        usb_location = _usb_location(os.path.join(path, name))

        devices.append(VideoDeviceInfo(device=device_num, name=model, is_boson=is_boson,
                                        usb_location=usb_location))

    return devices


def resolve_boson_video_devices(devices: List[VideoDeviceInfo]) -> List[VideoDeviceInfo]:
    """
    Collapse each physical Boson's multiple /dev/videoN nodes down to the one
    node that actually streams frames, mirroring flirpy's own disambiguation
    (flirpy.camera.boson.Boson.find_video_device).
    """
    import cv2

    groups: "dict[Optional[str], List[VideoDeviceInfo]]" = {}
    for device in devices:
        if not device.is_boson:
            continue
        groups.setdefault(device.physical_id, []).append(device)
    groups.pop(None, None)

    resolved = []
    for candidates in groups.values():
        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue

        for candidate in sorted(candidates, key=lambda d: d.device):
            cap = cv2.VideoCapture(candidate.device + cv2.CAP_V4L2)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                resolved.append(candidate)
                break

    return resolved


def print_report(ports: List[SerialPortInfo], devices: List[VideoDeviceInfo]) -> None:
    print("Serial (COM) ports:")
    if not ports:
        print("  (none found)")
    for port in ports:
        marker = " [Boson]" if port.is_boson else ""
        vidpid = f"{port.vid:04X}:{port.pid:04X}" if port.vid is not None else "----:----"
        hub = f"  (hub: {port.usb_location.root_hub}, port: {port.usb_location.port_path})" if port.usb_location else ""
        print(f"  {port.device:<15} {vidpid}  {port.description}{marker}{hub}")

    print("\nVideo devices:")
    if not devices:
        print("  (none found)")
    for device in devices:
        marker = " [Boson]" if device.is_boson else ""
        hub = f"  (hub: {device.usb_location.root_hub}, port: {device.usb_location.port_path})" if device.usb_location else ""
        print(f"  /dev/video{device.device:<5} {device.name}{marker}{hub}")

    boson_ports = [p for p in ports if p.is_boson]
    boson_devices = resolve_boson_video_devices(devices)
    if boson_ports or boson_devices:
        print("\nDetected Boson camera(s):")
        ports_by_physical_id = {p.physical_id: p for p in boson_ports}
        unmatched_devices = []
        for device in boson_devices:
            port = ports_by_physical_id.pop(device.physical_id, None)
            if port is not None:
                hub = f"  (hub: {device.usb_location.root_hub}, port: {device.usb_location.port_path})" \
                    if device.usb_location else ""
                print(f"  --camera {device.device}:{port.device}{hub}")
            else:
                unmatched_devices.append(device)

        if unmatched_devices or ports_by_physical_id:
            print("  Warning: couldn't pair every Boson video device with a "
                  "serial port; pair them manually.")
            for device in unmatched_devices:
                print(f"    Unpaired video device: /dev/video{device.device}")
            for port in ports_by_physical_id.values():
                print(f"    Unpaired serial port: {port.device}")


def main() -> None:
    ports = find_serial_ports()
    devices = find_video_devices()
    print_report(ports, devices)


if __name__ == "__main__":
    main()
