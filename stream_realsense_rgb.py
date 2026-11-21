"""Serve redirected RealSense RGB frames over HTTP MJPEG using FFmpeg."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import secrets
import subprocess
from threading import Condition, Thread


CAMERA = "Intel(R) RealSense(TM) Depth Camera 435if RGB (redirected)"


def capture_frames(pipe, ready, capture):
    try:
        for line in pipe:
            if not line.lower().startswith(b"content-length:"):
                continue
            size = int(line.split(b":", 1)[1])
            if pipe.readline() != b"\r\n":
                raise ValueError("Invalid FFmpeg MJPEG header")
            jpeg = pipe.read(size)
            if len(jpeg) != size:
                break
            with ready:
                capture["jpeg"] = jpeg
                capture["number"] += 1
                ready.notify_all()
    finally:
        with ready:
            capture["ended"] = True
            ready.notify_all()


def serve(bind, port, camera):
    path = f"/{secrets.token_urlsafe(24)}.mjpg"
    ready = Condition()
    capture = {"jpeg": None, "number": 0, "ended": False}

    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != path:
                self.send_error(404)
                return

            self.connection.settimeout(5)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                number = 0
                while True:
                    with ready:
                        updated = ready.wait_for(
                            lambda: capture["ended"] or capture["number"] != number,
                            timeout=10,
                        )
                        if not updated or capture["ended"]:
                            return
                        number = capture["number"]
                        jpeg = capture["jpeg"]
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
            except OSError:
                return

        def log_request(self, code="-", size="-"):
            pass  # The stream URL contains its access token.

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", "dshow", "-video_size", "640x480", "-framerate", "30",
        "-i", f"video={camera}",
        "-an", "-c:v", "mjpeg", "-q:v", "5", "-f", "mpjpeg", "pipe:1",
    ]
    with ThreadingHTTPServer((bind, port), StreamHandler) as server:
        server.timeout = 1
        with subprocess.Popen(command, stdout=subprocess.PIPE) as ffmpeg:
            reader = Thread(target=capture_frames, args=(ffmpeg.stdout, ready, capture))
            reader.start()
            try:
                with ready:
                    ready.wait_for(lambda: capture["number"] or capture["ended"], timeout=15)
                    if capture["ended"] or not capture["number"]:
                        raise SystemExit("FFmpeg did not capture a camera frame")
                print(f"RGB stream (PID {os.getpid()}): http://{bind}:{port}{path}", flush=True)
                while not capture["ended"]:
                    server.handle_request()
                raise SystemExit("FFmpeg camera stream ended")
            finally:
                if ffmpeg.poll() is None:
                    ffmpeg.terminate()
                reader.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="IPv4 address of the LAN interface")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", default=CAMERA, help="Exact DirectShow device name")
    options = parser.parse_args()
    try:
        serve(options.bind, options.port, options.camera)
    except KeyboardInterrupt:
        pass
