"""Portable Windows publisher for the RealSense RGB-D viewer."""

from contextlib import closing
from datetime import datetime
from http.client import HTTPConnection, HTTPException
from io import BytesIO
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4

from realsense_rgbd import native_camera_frames, pack_frame


WORKER_TIMEOUT = 18


class PublishError(Exception):
    pass


class PublishConnection:
    def __init__(self, destination):
        address = urlsplit(destination)
        self.connection = HTTPConnection(address.hostname, address.port, timeout=4)
        self.route = address.path
        self.headers = {
            "X-Publisher-ID": uuid4().hex,
            "Content-Type": "application/octet-stream",
        }

    def request(self, method, body=None):
        self.connection.request(method, self.route, body=body, headers=self.headers)
        with self.connection.getresponse() as response:
            response.read()
            if response.status == 409:
                raise PublishError("Another publisher is active, or the connection expired.")
            if response.status != 204:
                raise PublishError(f"The viewer returned HTTP {response.status} {response.reason}.")

    def close(self):
        self.connection.close()


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def encode_native_frame(color, depth, metadata):
    from PIL import Image

    rgb = Image.frombytes("RGB", (color.width, color.height), memoryview(color.get_data()).tobytes(), "raw", "BGR")
    z16 = Image.frombytes("I;16", (depth.width, depth.height), memoryview(depth.get_data()).tobytes())
    with BytesIO() as jpeg, BytesIO() as png:
        rgb.save(jpeg, format="JPEG", quality=90)
        z16.save(png, format="PNG", compress_level=1)
        return pack_frame(metadata, jpeg.getvalue(), png.getvalue())


def stream_camera(link, context, serial, usb_type, safe_profile):
    profiles = [(30, 30), (15, 15), (15, 30)]
    if safe_profile or usb_type.startswith(("1", "2")):
        profiles = profiles[1:]
    for depth_fps, color_fps in profiles:
        link.request("POST", b"")
        log(f"Opening RealSense at 640x480 (depth {depth_fps} fps, RGB {color_fps} fps).")
        with closing(native_camera_frames(depth_fps, serial, color_fps=color_fps, context=context)) as frames:
            try:
                first = next(frames)
            except RuntimeError as error:
                log(f"Camera could not start that profile: {error}")
                continue
            link.request("POST", b"")  # Startup can outlast the initial viewer claim.
            link.request("POST", encode_native_frame(*first))
            log("Streaming RGB + depth. Leave this window open. Press Ctrl+C to stop.")
            frame_count = 0
            last_report = monotonic()
            for frame in frames:
                link.request("POST", encode_native_frame(*frame))
                frame_count += 1
                now = monotonic()
                if now - last_report >= 10:
                    log(f"Sending RGB + depth at {frame_count / (now - last_report):.1f} fps.")
                    frame_count = 0
                    last_report = now
            return
    raise RuntimeError("Could not start paired RGB and depth capture. Check the USB connection and close other camera apps.")


def publish(config, safe_profile):
    log("Loading the RealSense SDK.")
    import pyrealsense2 as rs

    log("Initialising the RealSense USB backend.")
    context = rs.context('{"inherit":false,"dds":{"enabled":false}}')
    destination = urlsplit(config["destination"])
    waiting = False
    while True:
        if not waiting:
            log("Scanning for the USB RealSense.")
        else:
            print(flush=True)  # The supervisor treats blank lines as heartbeats.
        devices = context.query_devices()
        if not len(devices):
            if not waiting:
                log("Waiting for the USB RealSense. Plug it into this PC. Press Ctrl+C to stop.")
                waiting = True
            sleep(2)
            continue
        waiting = False
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        usb_info = rs.camera_info.usb_type_descriptor
        usb_type = device.get_info(usb_info) if device.supports(usb_info) else "unknown"
        firmware = device.get_info(rs.camera_info.firmware_version)
        log(f"Found {device.get_info(rs.camera_info.name)} (USB {usb_type}, firmware {firmware}).")
        log(f"Connecting to viewer at {destination.netloc}.")

        with closing(PublishConnection(config["destination"])) as link:
            claimed = False
            try:
                link.request("POST", b"")
                claimed = True
                log("Viewer connected. Preparing camera capture.")
                sleep(0.75)
                stream_camera(link, context, serial, usb_type, safe_profile)
            except (OSError, HTTPException, PublishError) as error:
                log(f"Viewer connection error: {error}. Retrying in 3 seconds.")
            except RuntimeError as error:
                log(f"Camera error: {error}. Retrying in 3 seconds.")
            finally:
                if claimed:
                    try:
                        link.request("DELETE")
                    except (OSError, HTTPException, PublishError):
                        pass  # The viewer releases expired connections automatically.
        sleep(3)


def run_camera_worker(command, timeout):
    lines = Queue()
    last_message = "camera worker startup"
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace") as worker:
        def collect():
            try:
                for line in worker.stdout:
                    lines.put(line.rstrip())
            finally:
                lines.put(None)

        reader = Thread(target=collect, daemon=True)
        reader.start()
        try:
            while (line := lines.get(timeout=timeout)) is not None:
                if line:
                    last_message = line.split("] ", 1)[-1]
                    print(line, flush=True)
            worker.wait(timeout=2)
            log(f"Camera process exited with code {worker.returncode}. Retrying in 3 seconds.")
            return None
        except (Empty, subprocess.TimeoutExpired):
            log(f"Camera stopped responding after: {last_message} Restarting capture in 3 seconds.")
            return last_message
        finally:
            if worker.poll() is None:
                try:
                    worker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait()
            reader.join(timeout=1)


def supervise():
    base_command = [sys.executable, "-B", "-u", "-X", "utf8", str(Path(__file__).resolve()), "--camera-worker"]
    safe_profile = False
    while True:
        command = base_command + (["--safe-profile"] if safe_profile else [])
        stalled_stage = run_camera_worker(command, WORKER_TIMEOUT)
        if stalled_stage and stalled_stage.startswith("Opening RealSense at 640x480 (depth 30 fps"):
            safe_profile = True
            log("Trying 15 fps on the next attempt.")
        sleep(3)


def worker_main():
    config = json.loads(Path(__file__).with_name("publisher-settings.json").read_text(encoding="utf-8"))
    try:
        publish(config, "--safe-profile" in sys.argv)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if "--camera-worker" in sys.argv:
        worker_main()
    else:
        log("RealSense RGB-D Publisher")
        try:
            supervise()
        except KeyboardInterrupt:
            log("Publisher stopped.")
