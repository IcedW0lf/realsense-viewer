"""Host the RealSense person viewer on a LAN interface."""

import argparse
from base64 import b64encode
from contextlib import closing
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from ipaddress import ip_address, ip_network
import json
import math
import os
from pathlib import Path
from shutil import copyfileobj
import subprocess
from threading import Condition, Event, Thread
from time import monotonic
from urllib.parse import urlsplit

import cv2
import numpy as np
import onnxruntime as ort

from realsense_rgbd import MAX_FRAME_SIZE, read_frame, read_frames
from stream_realsense_rgb import CAMERA


ASSETS = Path(__file__).resolve().with_name("viewer")
FRAME_LIFETIME = 3
PUBLISHER_LEASE = 10
DEPTH_RANGE = (0.2, 6.0)
OFFLINE = b'{"mode":"waiting"}'


class PersonDetector:
    """Run the official YOLOX-Tiny COCO ONNX model."""

    def __init__(self, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        input_layer = self.session.get_inputs()[0]
        self.input_name = input_layer.name
        _, _, self.height, self.width = input_layer.shape
        grids = []
        strides = []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(self.width // stride), np.arange(self.height // stride))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.append(np.full((x.size, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def detect(self, image):
        height, width = image.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized = cv2.resize(image, (int(width * scale), int(height * scale)))
        canvas = np.full((self.height, self.width, 3), 114, np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self.session.run(None, {self.input_name: blob})[0][0]

        scores = output[:, 4] * output[:, 5]  # COCO class 0 is person.
        keep = (scores >= 0.35) & (np.argmax(output[:, 5:], axis=1) == 0)
        centers = (output[keep, :2] + self.grid[keep]) * self.stride[keep] / scale
        sizes = np.exp(output[keep, 2:4]) * self.stride[keep] / scale
        low = np.clip(centers - sizes / 2, 0, (width - 1, height - 1))
        high = np.clip(centers + sizes / 2, 0, (width - 1, height - 1))
        scores = scores[keep]
        boxes = np.column_stack((low, high - low))
        picked = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.35, 0.45)
        return sorted(
            [(low[i].tolist() + high[i].tolist(), float(scores[i])) for i in picked],
            key=lambda item: item[0][0] + item[0][2],
        )


class DepthProjection:
    """Map sampled native depth pixels into the calibrated color camera."""

    step = 2

    def __init__(self, calibration):
        import pyrealsense2 as rs

        self.rs = rs
        self.color = self.make_intrinsics(calibration["color_intrinsics"])
        depth = self.make_intrinsics(calibration["depth_intrinsics"])
        if depth.model == rs.distortion.modified_brown_conrady:
            raise ValueError("The native depth profile cannot be deprojected")

        y, x = np.mgrid[:depth.height:self.step, :depth.width:self.step]
        if depth.model == rs.distortion.none or not any(depth.coeffs):
            self.rays = np.column_stack(((x.ravel() - depth.ppx) / depth.fx,
                                         (y.ravel() - depth.ppy) / depth.fy,
                                         np.ones(x.size))).astype(np.float32)
        else:
            pixels = np.column_stack((x.ravel(), y.ravel()))
            self.rays = np.array(
                [rs.rs2_deproject_pixel_to_point(depth, pixel, 1.0) for pixel in pixels.tolist()],
                np.float32,
            )
        transform = calibration["depth_to_color"]
        rotation = np.array(transform["rotation_column_major"], np.float32).reshape(3, 3, order="F")
        self.rays = self.rays @ rotation.T
        self.translation = np.array(transform["translation_m"], np.float32)

    def make_intrinsics(self, values):
        intrinsics = self.rs.intrinsics()
        for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
            setattr(intrinsics, key, values[key])
        intrinsics.model = self.rs.distortion.__members__[values["distortion_model"].split(".")[-1]]
        intrinsics.coeffs = values["coefficients"]
        return intrinsics

    def project(self, frame):
        z = frame.depth_raw[::self.step, ::self.step].ravel().astype(np.float32)
        z *= frame.metadata["depth_scale_m"]
        valid = (z >= 0.15) & (z <= 10)
        xyz = self.rays[valid] * z[valid, None] + self.translation
        xyz = xyz[xyz[:, 2] > 0]
        return self.project_color(xyz), xyz

    def project_color(self, xyz):
        intrinsics = self.color
        distortion = self.rs.distortion
        brown_models = (
            distortion.brown_conrady, distortion.modified_brown_conrady, distortion.inverse_brown_conrady,
        )
        if intrinsics.model not in (distortion.none, *brown_models):
            return np.array(
                [self.rs.rs2_project_point_to_pixel(intrinsics, point) for point in xyz.tolist()],
                np.float32,
            ).reshape(-1, 2)

        x, y = (xyz[:, :2] / xyz[:, 2, None]).T
        if intrinsics.model in brown_models:
            k1, k2, p1, p2, k3 = intrinsics.coeffs
            radius_sq = x * x + y * y
            radial = 1 + radius_sq * (k1 + radius_sq * (k2 + radius_sq * k3))
            scaled_x, scaled_y = radial * x, radial * y
            if intrinsics.model != distortion.brown_conrady:
                x, y = scaled_x, scaled_y
            xy = x * y
            x, y = (
                scaled_x + 2 * p1 * xy + p2 * (radius_sq + 2 * x * x),
                scaled_y + 2 * p2 * xy + p1 * (radius_sq + 2 * y * y),
            )
        return np.column_stack((x * intrinsics.fx + intrinsics.ppx, y * intrinsics.fy + intrinsics.ppy))

    def colorize(self, points):
        uv, xyz = points
        width = (self.color.width + self.step - 1) // self.step
        height = (self.color.height + self.step - 1) // self.step
        pixels = np.floor((uv + 0.5) / self.step)
        inside = ((pixels >= 0) & (pixels < (width, height))).all(axis=1)
        x, y = pixels[inside].astype(np.intp).T
        nearest = np.full((height, width), np.inf, np.float32)
        np.minimum.at(nearest, (y, x), xyz[inside, 2])

        # Fill small projection gaps for display. Position estimates use the raw points.
        neighbors = cv2.erode(nearest, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
        registered = np.where(np.isfinite(nearest), nearest, neighbors)
        near, far = DEPTH_RANGE
        shade = np.clip((far - registered) * (255 / (far - near)), 0, 255).astype(np.uint8)
        painted = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
        alpha = (np.isfinite(registered) * 255).astype(np.uint8)
        return np.dstack((painted, alpha))


def torso_position(points, anchor, width, height):
    if points is None:
        return None
    uv, xyz = points
    inside = (np.abs(uv[:, 0] - anchor[0]) < width * 0.32) & (np.abs(uv[:, 1] - anchor[1]) < height * 0.13)
    samples = xyz[inside]
    minimum = max(8, math.ceil(len(samples) * 0.22))
    if len(samples) < minimum:
        return None

    # Use the nearest depth band with enough support to reject isolated pixels.
    order = np.argsort(samples[:, 2])
    depth = samples[order, 2]
    stops = np.searchsorted(depth, depth + np.maximum(0.15, depth * 0.1), side="right")
    supported = np.flatnonzero(stops - np.arange(len(depth)) >= minimum)
    if not len(supported):
        return None
    start = supported[0]
    inliers = order[start:stops[start]]
    return np.median(samples[inliers], axis=0).tolist(), np.median(uv[inside][inliers], axis=0).tolist()


def locate_people(image, detections, points, intrinsics=None):
    height, width = image.shape[:2]
    if intrinsics:
        cx, cy = intrinsics.ppx, intrinsics.ppy
        fx, fy = intrinsics.fx, intrinsics.fy
    else:
        # Nominal RGB field of view of the RealSense D435if. RDP cropping may differ.
        cx, cy = (width - 1) / 2, (height - 1) / 2
        fx, fy = cx / math.tan(math.radians(69 / 2)), cy / math.tan(math.radians(42 / 2))

    people = []
    for box, score in detections:
        left, top, right, bottom = box
        anchor = ((left + right) / 2, top + (bottom - top) * 0.38)
        measured = torso_position(points, anchor, right - left, bottom - top)
        if measured is None:
            position = None
            direction = [(anchor[0] - cx) / fx, (anchor[1] - cy) / fy, 1.0]
        else:
            position, anchor = measured
            direction = position
            camera_x, camera_y, camera_z = position
            # Convert RealSense optical coordinates to a right-handed Z-up frame.
            position = [camera_x, camera_z, -camera_y]
        people.append({
            "box": box,
            "anchor": anchor,
            "score": score,
            "bearing": math.degrees(math.atan2(direction[0], direction[2])),
            "position": position,
            "range": math.hypot(*position) if position else None,
        })
    return people, (cx, cy)


def rgb_frames(url):
    camera = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
    ])
    try:
        while camera.isOpened():
            ok, image = camera.read()
            if not ok:
                return
            yield image, None
    finally:
        camera.release()


def redirected_frames(camera, viewer):
    width, height = 640, 480
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "dshow", "-video_size", f"{width}x{height}", "-framerate", "30",
        "-i", f"video={camera}", "-an", "-threads:v", "1", "-c:v", "rawvideo",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    with viewer.ready:
        if viewer.publisher_active():
            return
        ffmpeg = subprocess.Popen(command, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        viewer.rdp_process = ffmpeg
    try:
        with ffmpeg:
            size = width * height * 3
            try:
                while True:
                    pixels = ffmpeg.stdout.read(size)
                    if len(pixels) != size:
                        return
                    yield np.frombuffer(pixels, np.uint8).reshape(height, width, 3), None
            finally:
                if ffmpeg.poll() is None:
                    ffmpeg.terminate()
    finally:
        with viewer.ready:
            viewer.rdp_process = None


def rgbd_frames(url, token):
    with closing(read_frames(url, token, timeout=3)) as source:
        for frame in source:
            yield frame.color_bgr, frame


class LiveViewer:
    def __init__(self, detector):
        self.detector = detector
        self.ready = Condition()
        self.stop = Event()
        self.inputs = {}
        self.output = (0, OFFLINE)
        self.publisher = None
        self.rdp_process = None

    def publisher_active(self):
        return self.publisher is not None and monotonic() - self.publisher[1] < PUBLISHER_LEASE

    def claim_publisher(self, identifier):
        with self.ready:
            active = self.publisher_active()
            if active and self.publisher[0] != identifier:
                return False
            self.publisher = (identifier, monotonic())
            if not active:
                self.inputs.pop("rgb", None)
                if self.rdp_process is not None:
                    self.rdp_process.terminate()
                print("Portable publisher connected; waiting for RGB-D frames.", flush=True)
                self.ready.notify_all()
            return True

    def owns_publisher(self, identifier):
        return self.publisher_active() and self.publisher[0] == identifier

    def submit_publisher_frame(self, identifier, frame):
        with self.ready:
            if not self.owns_publisher(identifier):
                return False
            now = monotonic()
            self.publisher = (identifier, now)
            if "portable" not in self.inputs:
                print("Receiving RGB-D frames from the portable publisher.", flush=True)
            self.inputs["portable"] = (now, frame.color_bgr, frame)
            self.ready.notify_all()
            return True

    def release_publisher(self, identifier):
        with self.ready:
            if self.publisher is not None and self.publisher[0] == identifier:
                self.publisher = None
                self.inputs.pop("portable", None)
                self.output = (0, OFFLINE)
                print("Portable publisher disconnected.", flush=True)
                self.ready.notify_all()

    def capture(self, mode, source, *args):
        while not self.stop.is_set():
            try:
                with closing(source(*args)) as frames:
                    for image, depth in frames:
                        if self.stop.is_set():
                            return
                        with self.ready:
                            if mode == "rgb" and self.publisher_active():
                                continue
                            self.inputs[mode] = (monotonic(), image, depth)
                            self.ready.notify()
            except (EOFError, HTTPException, OSError, ValueError, RuntimeError, cv2.error):
                pass  # A dropped connection is retried without blocking the other source.
            self.stop.wait(1)

    def process(self):
        last_frame = None
        last_output = 0
        frame_interval = None
        projection = None
        calibration = None
        sequence = 0
        while not self.stop.is_set():
            with self.ready:
                now = monotonic()
                frame = self.inputs.get("portable") if self.publisher_active() else None
                if frame is None or now - frame[0] > 1:
                    frame = self.inputs.get("rgbd")
                if frame is None or now - frame[0] > 1:
                    frame = self.inputs.get("rgb") if not self.publisher_active() else None
                if frame is None or frame is last_frame or monotonic() - frame[0] > FRAME_LIFETIME:
                    self.ready.wait(0.2)
                    continue
            last_frame = frame
            received, image, depth = frame
            detections = self.detector.detect(image)
            points = None
            depth_image = None
            if depth is not None:
                current = {key: depth.metadata[key] for key in (
                    "color_intrinsics", "depth_intrinsics", "depth_to_color",
                )}
                if current != calibration:
                    projection = DepthProjection(current)
                    calibration = current
                points = projection.project(depth)
                depth_ok, depth_png = cv2.imencode(".png", projection.colorize(points), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if depth_ok:
                    depth_image = b64encode(depth_png).decode("ascii")

            people, origin = locate_people(image, detections, points, projection.color if depth is not None else None)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            sequence += 1
            now = monotonic()
            if last_output:
                interval = now - last_output
                frame_interval = interval if frame_interval is None else 0.85 * frame_interval + 0.15 * interval
            packet = json.dumps({
                "sequence": sequence,
                "mode": "rgbd" if depth is not None else "rgb",
                "fps": round(1 / frame_interval, 1) if frame_interval else 0,
                "width": image.shape[1], "height": image.shape[0],
                "image": b64encode(jpeg).decode("ascii"),
                "depth_image": depth_image,
                "depth_range": DEPTH_RANGE if depth_image else None,
                "origin": origin,
                "position_axes": {"x": "right", "y": "forward", "z": "up"},
                "people": people,
            }, allow_nan=False).encode("utf-8")
            self.output = (received, packet)
            last_output = now

    def snapshot(self):
        received, packet = self.output
        return packet if monotonic() - received <= FRAME_LIFETIME else OFFLINE


def make_handler(viewer, publisher_config=None, allowed_networks=()):
    downloads = {}
    if publisher_config:
        downloads["/download/RealSense-Publisher-Portable.zip"] = Path(publisher_config["portable_artifact"])
        downloads["/download/publisher-settings.json"] = Path(publisher_config["settings_artifact"])
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    }

    class ViewerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def parse_request(self):
            if not super().parse_request():
                return False
            peer = ip_address(self.client_address[0])
            if allowed_networks and not any(peer in network for network in allowed_networks):
                self.send_error(403)
                return False
            return True

        def send_bytes(self, body, media_type):
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in downloads:
                self.download_publisher(downloads[path])
                return
            if path == "/api/frame":
                body, media_type = viewer.snapshot(), "application/json"
            elif path in assets:
                filename, media_type = assets[path]
                body = (ASSETS / filename).read_bytes()
            else:
                self.send_error(404)
                return

            try:
                self.send_bytes(body, media_type)
            except OSError:
                return

        def download_publisher(self, artifact):
            try:
                package = artifact.open("rb")
            except OSError:
                self.send_error(404)
                return
            with package:
                try:
                    self.send_response(200)
                    media_type = "application/zip" if artifact.suffix == ".zip" else "application/json"
                    self.send_header("Content-Type", media_type)
                    self.send_header("Content-Disposition", f'attachment; filename="{artifact.name}"')
                    self.send_header("Content-Length", str(os.fstat(package.fileno()).st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    copyfileobj(package, self.wfile)
                except OSError:
                    return

        def publisher_identifier(self):
            if not publisher_config or self.path != "/api/publish":
                self.send_error(404)
                return None
            identifier = self.headers.get("X-Publisher-ID", "")
            if len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):
                self.send_error(400)
                return None
            return identifier

        def do_POST(self):
            identifier = self.publisher_identifier()
            if identifier is None:
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= size <= MAX_FRAME_SIZE:
                self.send_error(413)
                return
            if size == 0:
                accepted = viewer.claim_publisher(identifier)
            else:
                with viewer.ready:
                    owns_lease = viewer.owns_publisher(identifier)
                if not owns_lease:
                    self.send_error(409)
                    return
                try:
                    data = BytesIO(self.rfile.read(size))
                    frame = read_frame(data)
                    if frame is None or data.tell() != size:
                        raise ValueError("Invalid frame length")
                except (EOFError, OSError, ValueError, cv2.error):
                    self.send_error(400)
                    return
                accepted = viewer.submit_publisher_frame(identifier, frame)
            if not accepted:
                self.send_error(409)
                return
            self.send_response(204)
            self.end_headers()

        def do_DELETE(self):
            identifier = self.publisher_identifier()
            if identifier is not None:
                viewer.release_publisher(identifier)
                self.send_response(204)
                self.end_headers()

        def log_message(self, *args):
            pass

    return ViewerHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="IPv4 address of the LAN interface")
    parser.add_argument("--allow-network", action="append", type=ip_network, default=[], help="Network allowed to reach the viewer")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--model", required=True, type=Path, help="Official YOLOX-Tiny ONNX weights")
    parser.add_argument("--rgb-url", default=os.getenv("REALSENSE_VIEWER_RGB_URL"))
    parser.add_argument("--rgbd-url", default=os.getenv("REALSENSE_VIEWER_RGBD_URL"))
    parser.add_argument("--rdp-camera", nargs="?", const=CAMERA, help="Capture from a Windows redirected DirectShow camera")
    parser.add_argument("--publisher-config", type=Path, help="Receive RGB-D from the portable publisher")
    args = parser.parse_args()
    if not (args.rdp_camera or args.rgb_url or args.rgbd_url or args.publisher_config):
        parser.error("Configure a camera source or --publisher-config")
    depth_token = os.getenv("REALSENSE_RGBD_TOKEN")
    if args.rgbd_url and not depth_token:
        parser.error("Set REALSENSE_RGBD_TOKEN for the RGB-D sender")
    publisher_config = json.loads(args.publisher_config.read_text(encoding="utf-8")) if args.publisher_config else None
    if args.rgbd_url or publisher_config:
        import pyrealsense2  # Check the optional RGB-D dependency before listening.

    viewer = LiveViewer(PersonDetector(args.model))
    with ThreadingHTTPServer((args.bind, args.port), make_handler(viewer, publisher_config, args.allow_network)) as server:
        if args.rdp_camera:
            Thread(target=viewer.capture, args=("rgb", redirected_frames, args.rdp_camera, viewer), daemon=True).start()
        elif args.rgb_url:
            Thread(target=viewer.capture, args=("rgb", rgb_frames, args.rgb_url), daemon=True).start()
        if args.rgbd_url:
            Thread(target=viewer.capture, args=("rgbd", rgbd_frames, args.rgbd_url, depth_token), daemon=True).start()
        Thread(target=viewer.process, daemon=True).start()
        print(f"Viewer URL: http://{args.bind}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            viewer.stop.set()


if __name__ == "__main__":
    main()
