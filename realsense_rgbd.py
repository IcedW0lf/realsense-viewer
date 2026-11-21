"""Stream RealSense RGB and metric depth from a USB host to Python clients.

On the USB host, install pyrealsense2, numpy and opencv-python-headless.
Run `python realsense_rgbd.py serve --bind <USB_HOST_LAN_IP>`.

On each reader, install numpy and opencv-python-headless. Run
`python realsense_rgbd.py read http://<USB_HOST_LAN_IP>:8766/rgbd` and enter
the printed token. For computation, import `read_frames(url, token)`.

Each frame contains OpenCV BGR, native uint16 Z16 depth and calibration.
Depth and color have independent pixel coordinates. Use the calibration in
frame.metadata to register them. A raw depth value of zero is invalid.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from getpass import getpass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
from struct import Struct
from threading import Condition, Thread
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import numpy as np


STREAM_PATH = "/rgbd"
STREAM_MAGIC = b"RSRGBD\x01"
# Network-order byte counts for JSON metadata, JPEG color and 16-bit PNG depth.
FRAME_HEADER = Struct("!III")
MAX_METADATA_SIZE = 65_536
MAX_IMAGE_SIZE = 4_194_304
MAX_FRAME_SIZE = FRAME_HEADER.size + MAX_METADATA_SIZE + 2 * MAX_IMAGE_SIZE


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    metadata: dict

    @property
    def depth_m(self):
        """Depth in metres along the depth camera's optical axis. Zero is invalid."""
        import numpy as np

        return self.depth_raw.astype(np.float32) * self.metadata["depth_scale_m"]


def intrinsics(profile):
    model = profile.get_intrinsics()
    return {
        "width": model.width, "height": model.height,
        "fx": model.fx, "fy": model.fy, "ppx": model.ppx, "ppy": model.ppy,
        "distortion_model": str(model.model), "coefficients": list(model.coeffs),
    }


def frame_timing(frame):
    return {
        "frame_number": frame.get_frame_number(),
        "timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
    }


def native_camera_frames(fps, serial, *, color_fps=None, context=None):
    import pyrealsense2 as rs

    if context is None:
        context = rs.context()
        if not len(context.query_devices()):
            raise RuntimeError("RealSense SDK found no camera. Run the sender on the USB host.")

    pipeline = rs.pipeline(context)
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, color_fps or fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    active = pipeline.start(config)
    try:
        color = active.get_stream(rs.stream.color).as_video_stream_profile()
        depth = active.get_stream(rs.stream.depth).as_video_stream_profile()
        extrinsics = depth.get_extrinsics_to(color)
        calibration = {
            "color_intrinsics": intrinsics(color),
            "depth_intrinsics": intrinsics(depth),
            "depth_to_color": {
                "rotation_column_major": list(extrinsics.rotation),
                "translation_m": list(extrinsics.translation),
            },
        }
        deadline = monotonic() + 5
        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                if monotonic() >= deadline:
                    raise RuntimeError("Camera did not supply a complete RGB and depth frame pair")
                continue
            yield (
                color_frame,
                depth_frame,
                {
                    **calibration,
                    "depth_scale_m": depth_frame.get_units(),
                    "color_timing": frame_timing(color_frame),
                    "depth_timing": frame_timing(depth_frame),
                },
            )
            deadline = monotonic() + 5
    finally:
        pipeline.stop()


def camera_frames(fps, serial):
    import numpy as np

    with closing(native_camera_frames(fps, serial)) as source:
        for color, depth, metadata in source:
            yield RGBDFrame(np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), metadata)


def pack_frame(metadata, color, depth):
    metadata = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return b"".join((FRAME_HEADER.pack(len(metadata), len(color), len(depth)), metadata, color, depth))


def encode_frame(frame):
    import cv2

    color_ok, color = cv2.imencode(".jpg", frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    depth_ok, depth = cv2.imencode(".png", frame.depth_raw, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not color_ok or not depth_ok:
        raise RuntimeError("Could not encode the camera frame")
    return pack_frame(frame.metadata, color, depth)


def serve(source, bind, port, token):
    authorization = f"Bearer {token}".encode("utf-8")
    ready = Condition()
    sequence = 0
    packet = None
    closed = False

    class StreamHandler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def do_GET(self):
            if self.path != STREAM_PATH:
                self.send_error(404)
                return
            supplied = self.headers.get("Authorization", "").encode("utf-8")
            if not secrets.compare_digest(authorization, supplied):
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                self.end_headers()
                return

            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(STREAM_MAGIC)
                last_sequence = 0
                while True:
                    with ready:
                        available = ready.wait_for(lambda: closed or sequence != last_sequence, 10)
                        if closed or not available:
                            return
                        last_sequence, latest_packet = sequence, packet
                    self.wfile.write(latest_packet)
            except OSError:
                return

    with ThreadingHTTPServer((bind, port), StreamHandler) as server:
        packet = encode_frame(next(source))
        sequence = 1
        listener = Thread(target=server.serve_forever)
        listener.start()
        try:
            print(f"Stream URL: http://{bind}:{server.server_port}{STREAM_PATH}", flush=True)
            print(f"Access token: {token}", flush=True)
            for frame in source:
                next_packet = encode_frame(frame)
                with ready:
                    packet = next_packet
                    sequence += 1
                    ready.notify_all()
        finally:
            with ready:
                closed = True
                ready.notify_all()
            server.shutdown()
            listener.join()


def read_exact(response, size):
    data = response.read(size)
    if len(data) != size:
        raise EOFError("The sender disconnected in the middle of a frame")
    return data


def read_frame(stream):
    import cv2
    import numpy as np

    header = stream.read(FRAME_HEADER.size)
    if not header:
        return None
    if len(header) != FRAME_HEADER.size:
        raise EOFError("The sender disconnected in the frame header")
    metadata_size, color_size, depth_size = FRAME_HEADER.unpack(header)
    if not (0 < metadata_size <= MAX_METADATA_SIZE and 0 < color_size <= MAX_IMAGE_SIZE
            and 0 < depth_size <= MAX_IMAGE_SIZE):
        raise ValueError("Invalid RGB-D frame lengths")
    metadata = json.loads(read_exact(stream, metadata_size))
    color_jpeg = np.frombuffer(read_exact(stream, color_size), np.uint8)
    depth_png = np.frombuffer(read_exact(stream, depth_size), np.uint8)
    color = cv2.imdecode(color_jpeg, cv2.IMREAD_COLOR)
    depth = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED)
    if color is None or depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Invalid color or 16-bit depth image")
    return RGBDFrame(color, depth, metadata)


def read_frames(url, token, timeout=10):
    """Yield RGBDFrame objects from the USB host. Close the iterator when finished."""
    address = urlsplit(url)
    if address.scheme != "http" or not address.hostname:
        raise ValueError("Use the sender's http://host:port/rgbd URL")
    with closing(HTTPConnection(address.hostname, address.port, timeout=timeout)) as connection:
        connection.request("GET", address.path or STREAM_PATH, headers={"Authorization": f"Bearer {token}"})
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"Sender returned HTTP {response.status} {response.reason}")
            if read_exact(response, len(STREAM_MAGIC)) != STREAM_MAGIC:
                raise ValueError("The URL did not return a compatible RealSense RGB-D stream")
            while (frame := read_frame(response)) is not None:
                yield frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sender = commands.add_parser("serve", help="Publish frames from the USB camera")
    sender.add_argument("--bind", required=True, help="USB host's LAN IPv4 address")
    sender.add_argument("--port", type=int, default=8766)
    sender.add_argument("--fps", type=int, choices=(15, 30), default=30)
    sender.add_argument("--serial", help="Camera serial number when several cameras are attached")
    reader = commands.add_parser("read", help="Check that the sender delivers color and depth")
    reader.add_argument("url")
    reader.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    token = os.environ.get("REALSENSE_RGBD_TOKEN")
    if args.command == "serve":
        with closing(camera_frames(args.fps, args.serial)) as source:
            serve(source, args.bind, args.port, token or secrets.token_urlsafe(24))
        return

    if args.frames < 1:
        parser.error("--frames must be at least 1")
    import numpy as np

    token = token or getpass("Access token: ")
    with closing(read_frames(args.url, token)) as frames:
        start = monotonic()
        for _ in range(args.frames):
            frame = next(frames, None)
            if frame is None:
                raise EOFError("The sender disconnected before the frame check finished")
    print(f"Received {args.frames} frame pairs in {monotonic() - start:.2f} s")
    print(f"Color {frame.color_bgr.shape}, depth {frame.depth_raw.shape} ({frame.depth_raw.dtype})")
    print(f"Depth scale {frame.metadata['depth_scale_m']:.6g} m/unit, {np.count_nonzero(frame.depth_raw)} valid pixels")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (EOFError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
