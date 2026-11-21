# RealSense RGB-D Viewer

A LAN dashboard that aligns RealSense depth to RGB, detects people with YOLOX-Tiny, and plots camera-relative positions in metres. Position axes are **X right, Y forward, Z up**.

## Start the viewer

Run the following on the Windows compute workstation with Python 3.13 installed. Replace the example IP and network with the workstation's LAN IPv4 address and the network intended to access the feed.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-viewer.txt
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx -OutFile .\models\yolox_tiny.onnx
.\.venv\Scripts\python.exe .\build_portable.py http://192.168.1.42:8767/
.\.venv\Scripts\python.exe .\realsense_viewer.py --bind 192.168.1.42 --allow-network 192.168.1.0/24 --model .\models\yolox_tiny.onnx --publisher-config .\dist\publisher-config.json
```

Open the configured viewer URL in a browser. The viewer and publishing endpoint accept connections from the configured LAN without a sign-in.

## Start the publisher

On the 64-bit Windows PC with the RealSense plugged into USB, open **Local setup** in the viewer and download the portable publisher. Extract the ZIP, then double-click **Start-RealSense.bat**. The publisher bundles its own Python runtime. Leave the console open while it shows **Streaming RGB + depth**. Camera redirection in RDP can be disabled so the local publisher can use the USB camera directly.

Select **RGB**, **RGB + Depth**, or **Depth** in the browser. Press **Y** to toggle the depth view, or drag the **3D** plot to rotate it. **Local setup** also provides the standalone `publisher-settings.json` for an existing publisher folder.
