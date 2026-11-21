"""Build the Windows x64 RealSense publisher ZIP and its viewer configuration."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE = Path(__file__).resolve().parent
PYTHON_RUNTIME_URL = "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-amd64.zip"
PYTHON_RUNTIME_HASH = "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38"
PUBLISHER_FILES = ("Start-RealSense.bat", "realsense_publisher.py", "realsense_rgbd.py")


def build_publisher(origin, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "RealSense-Publisher-Portable.zip"
    settings_path = output / "publisher-settings.json"
    viewer_config = output / "publisher-config.json"
    settings = {"destination": f"{origin}/api/publish", "viewer_url": f"{origin}/"}

    with urlopen(PYTHON_RUNTIME_URL, timeout=45) as response:
        binary = response.read()
    if sha256(binary).hexdigest() != PYTHON_RUNTIME_HASH:
        raise RuntimeError("Embedded Python checksum mismatch")

    with TemporaryDirectory(prefix="realsense-build-", dir=output) as temporary:
        stage = Path(temporary).resolve()
        folder = stage / "RealSense-Publisher"
        runtime = folder / "runtime"
        with ZipFile(BytesIO(binary)) as archive:
            archive.extractall(runtime)
        (runtime / "python313._pth").write_text("python313.zip\n.\nLib\\site-packages\n..\nimport site\n", encoding="ascii")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--no-deps", "--no-compile",
            "--platform", "win_amd64", "--implementation", "cp", "--python-version", "3.13", "--abi", "cp313",
            "--target", str(runtime / "Lib" / "site-packages"), "pyrealsense2==2.59.2.11152", "Pillow==13.0.0",
        ], check=True)
        for filename in PUBLISHER_FILES:
            copy2(SOURCE / filename, folder / filename)
        (folder / settings_path.name).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

        staged_zip = stage / artifact.name
        with ZipFile(staged_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(stage))
        staged_zip.replace(artifact)
        copy2(folder / settings_path.name, settings_path)

    config = {"portable_artifact": str(artifact), "settings_artifact": str(settings_path)}
    viewer_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(artifact)
    print(viewer_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_url", help="Viewer origin, such as http://192.168.1.42:8767/")
    parser.add_argument("--output", type=Path, default=SOURCE / "dist")
    args = parser.parse_args()
    address = urlsplit(args.viewer_url)
    if (address.scheme != "http" or not address.hostname or address.username is not None
            or address.path not in ("", "/") or address.query or address.fragment):
        parser.error("Use an HTTP viewer origin without a path, credentials, query or fragment")
    build_publisher(f"{address.scheme}://{address.netloc}", args.output)


if __name__ == "__main__":
    main()
