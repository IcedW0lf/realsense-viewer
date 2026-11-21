"use strict";

const byId = (id) => document.getElementById(id);
const video = byId("video");
const plot = byId("plot");
const videoContext = video.getContext("2d");
const plotContext = plot.getContext("2d");
const colors = ["#5ce6c2", "#ffbf82", "#b5a5ff", "#82c9ff", "#cee577", "#f5a1c0"];
const colorAt = (index) => colors[index % colors.length];
const signed = (value, places) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(places)}`;
const modeButtons = Array.from(document.querySelectorAll("[data-camera-mode]"));
const setupDialog = byId("setup");
const defaultOrbit = { yaw: -0.45, pitch: 0.62 };

let frame = null;
let image = null;
let depthImage = null;
let paused = false;
let overlays = true;
let cameraMode = "rgb";
let colorMode = "rgb";
let plotMode = "space";
let orbit = { ...defaultOrbit };
let drag = null;

function arrow(ctx, x1, y1, x2, y2, color, dashed = false) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  if (Math.hypot(x2 - x1, y2 - y1) < 4) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.8;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 9 * Math.cos(angle - 0.5), y2 - 9 * Math.sin(angle - 0.5));
  ctx.lineTo(x2 - 9 * Math.cos(angle + 0.5), y2 - 9 * Math.sin(angle + 0.5));
  ctx.closePath();
  ctx.fill();
}

function drawVideo() {
  if (!frame || !image) return;
  if (video.width !== frame.width || video.height !== frame.height) {
    video.width = frame.width;
    video.height = frame.height;
    video.style.aspectRatio = `${frame.width} / ${frame.height}`;
  }
  const ctx = videoContext;
  const view = depthImage ? cameraMode : "rgb";
  if (view === "depth") {
    ctx.fillStyle = "#0a1219";
    ctx.fillRect(0, 0, video.width, video.height);
  } else {
    ctx.drawImage(image, 0, 0);
  }
  if (view !== "rgb") {
    ctx.save();
    ctx.globalAlpha = view === "blend" ? 0.48 : 1;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(depthImage, 0, 0, video.width, video.height);
    ctx.restore();
  }
  if (!overlays) return;
  frame.people.forEach((person, index) => {
    const color = colorAt(index);
    const [x1, y1, x2, y2] = person.box;
    const [ax, ay] = person.anchor;
    const [ox, oy] = frame.origin;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    arrow(ctx, ox, oy, ax, ay, color, !person.position);
    ctx.fillStyle = color;
    ctx.fillRect(ax - 2, ay - 2, 4, 4);
    const message = person.range === null
      ? `Person ${index + 1}  ${signed(person.bearing, 0)}°`
      : `Person ${index + 1}  ${person.range.toFixed(2)} m`;
    ctx.font = "600 13px 'Segoe UI', sans-serif";
    ctx.textBaseline = "middle";
    const labelWidth = ctx.measureText(message).width + 12;
    const labelLeft = Math.min(x1 - 1, frame.width - labelWidth);
    const labelTop = y1 > 23 ? y1 - 22 : y1 + 1;
    ctx.fillStyle = color;
    ctx.fillRect(labelLeft, labelTop, labelWidth, 20);
    ctx.fillStyle = "#0b1921";
    ctx.fillText(message, labelLeft + 6, labelTop + 10);
  });
}

function cameraViewControls() {
  const active = depthImage ? cameraMode : "rgb";
  modeButtons.forEach((button) => {
    const mode = button.dataset.cameraMode;
    button.disabled = mode !== "rgb" && !depthImage;
    button.setAttribute("aria-pressed", String(mode === active));
  });
  byId("depth-legend").hidden = active === "rgb";
  if (frame?.depth_range) {
    byId("depth-near").textContent = `${frame.depth_range[0]} m`;
    byId("depth-far").textContent = `${frame.depth_range[1]} m+`;
  }
}

function setCameraMode(mode) {
  if (mode !== "rgb" && !depthImage) return;
  cameraMode = mode;
  if (mode !== "depth") colorMode = mode;
  cameraViewControls();
  drawVideo();
}

function drawTopPlot(people, metric) {
  const ctx = plotContext;
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  const w = plot.width / 2;
  const h = plot.height / 2;
  const x = w / 2;
  const y = h - 34;
  const measured = people.flatMap((person) => person.position ? [person.position[1], Math.abs(person.position[0]) * 1.4] : []);
  const extent = Math.ceil(Math.max(3, ...measured) / 3) * 3;
  const radius = 170;
  ctx.clearRect(0, 0, w, h);

  ctx.textBaseline = "middle";
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillStyle = "#a6bac3";
  ctx.textAlign = "center";
  ctx.fillText(metric ? "Y · FORWARD" : "FORWARD", x, 20);
  ctx.textAlign = "left";
  ctx.fillText(metric ? "−X" : "LEFT", 16, y + 1);
  ctx.textAlign = "right";
  ctx.fillText(metric ? "+X" : "RIGHT", w - 16, y + 1);
  ctx.strokeStyle = "#33505d";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 35);
  ctx.lineTo(x, y);
  ctx.moveTo(47, y);
  ctx.lineTo(w - 47, y);
  ctx.stroke();
  ctx.setLineDash([]);

  for (let line = 1; line <= 3; line += 1) {
    const level = radius * line / 3;
    ctx.strokeStyle = "#243b49";
    ctx.beginPath();
    if (metric) {
      ctx.moveTo(43, y - level);
      ctx.lineTo(w - 35, y - level);
      ctx.fillStyle = "#a6bac3";
      ctx.textAlign = "left";
      ctx.fillText(`${extent * line / 3} m`, 13, y - level);
    } else {
      ctx.arc(x, y, level, Math.PI * 1.28, Math.PI * 1.72);
    }
    ctx.stroke();
  }

  people.forEach((person, index) => {
    const angle = person.bearing * Math.PI / 180;
    const color = colorAt(index);
    const point = person.position || [Math.sin(angle) * extent * 0.76, Math.cos(angle) * extent * 0.76, 0];
    const endX = x + point[0] * radius / extent;
    const endY = y - point[1] * radius / extent;
    arrow(ctx, x, y, endX, endY, color, !person.position);
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.font = "600 12px 'Segoe UI', sans-serif";
    ctx.fillText(`P${index + 1}`, endX, endY - 11);
  });
  ctx.fillStyle = "#5ce6c2";
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.textAlign = "center";
  ctx.fillStyle = "#c1d1da";
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillText("Camera", x, h - 16);
}

function drawSpacePlot(people) {
  const ctx = plotContext;
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  const width = plot.width / 2;
  const height = plot.height / 2;
  ctx.clearRect(0, 0, width, height);
  const measured = people.flatMap(({ position }) => position ? [Math.hypot(...position)] : []);
  const extent = Math.ceil(Math.max(2, ...measured) / 2) * 2;
  const half = extent * 0.58;
  const vectors = people.map((person, index) => {
    const angle = person.bearing * Math.PI / 180;
    return {
      position: person.position || [Math.sin(angle) * extent * 0.72, Math.cos(angle) * extent * 0.72, 0],
      measured: Boolean(person.position),
      color: colorAt(index),
      label: `P${index + 1}`,
    };
  });
  const yawSin = Math.sin(orbit.yaw);
  const yawCos = Math.cos(orbit.yaw);
  const pitchSin = Math.sin(orbit.pitch);
  const pitchCos = Math.cos(orbit.pitch);
  const rotate = ([x, y, z]) => [x * yawCos - y * yawSin, -z * pitchCos - (x * yawSin + y * yawCos) * pitchSin];
  const bounds = [
    [-half, 0, 0], [half, 0, 0], [-half, extent, 0], [half, extent, 0],
    [0, 0, half * 0.66], ...vectors.map(({ position }) => position),
  ].map(rotate);
  const xs = bounds.map(([x]) => x);
  const ys = bounds.map(([, y]) => y);
  const left = Math.min(...xs), right = Math.max(...xs);
  const top = Math.min(...ys), bottom = Math.max(...ys);
  const scale = Math.min((width - 74) / (right - left), (height - 70) / (bottom - top));
  const project = (point) => {
    const [x, y] = rotate(point);
    return [width / 2 + (x - (left + right) / 2) * scale, height / 2 + (y - (top + bottom) / 2) * scale];
  };

  ctx.strokeStyle = "#243b49";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = extent * tick / 4;
    const x = half * (tick / 2 - 1);
    ctx.moveTo(...project([-half, y, 0]));
    ctx.lineTo(...project([half, y, 0]));
    ctx.moveTo(...project([x, 0, 0]));
    ctx.lineTo(...project([x, extent, 0]));
  }
  ctx.stroke();

  const origin = project([0, 0, 0]);
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const axes = [
    [[half, 0, 0], "X", "#f7ab91"],
    [[0, extent, 0], "Y", "#8fc6ff"],
    [[0, 0, half * 0.66], "Z", "#f0d17b"],
  ];
  for (const [point, label, color] of axes) {
    const end = project(point);
    arrow(ctx, ...origin, ...end, color);
    ctx.fillText(label, end[0] + 9, end[1] + 8);
  }
  ctx.fillStyle = "#a6bac3";
  const midway = project([half, extent / 2, 0]);
  ctx.fillText(`${extent / 2} m`, midway[0] + 12, midway[1] - 11);

  for (const { position, measured, color, label } of vectors) {
    const end = project(position);
    arrow(ctx, ...origin, ...end, color, !measured);
    ctx.fillStyle = color;
    ctx.fillText(label, end[0], end[1] - 10);
  }
  ctx.fillStyle = "#5ce6c2";
  ctx.beginPath();
  ctx.arc(...origin, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#c1d1da";
  ctx.fillText("Camera", origin[0] - 8, origin[1] + 15);
}

function drawPlot() {
  const metric = frame?.mode === "rgbd";
  const space = metric && plotMode === "space";
  byId("plot-space").disabled = !metric;
  byId("plot-space").setAttribute("aria-pressed", String(space));
  byId("plot-top").setAttribute("aria-pressed", String(!space));
  byId("reset-orbit").hidden = !space;
  plot.dataset.rotatable = String(space);
  byId("plot-note").textContent = metric
    ? (space ? "X right · Y forward · Z up" : "Metres from camera · Dashed rays lack valid depth")
    : "Estimated bearings · Distance requires depth";
  if (space) drawSpacePlot(frame.people);
  else drawTopPlot(frame?.people || [], metric);
}

function renderDetections(people) {
  byId("count").textContent = String(people.length);
  byId("empty").hidden = people.length > 0;
  byId("empty").textContent = "No people in frame.";
  const rows = people.map((person, index) => {
    const li = document.createElement("li");
    li.className = "person";
    li.style.setProperty("--person", colorAt(index));
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `Person ${index + 1}`;
    const direction = document.createElement("span");
    direction.className = "direction";
    direction.textContent = person.range === null
      ? `${signed(person.bearing, 1)}° bearing`
      : `${person.range.toFixed(2)} m · ${signed(person.bearing, 1)}°`;
    const coords = document.createElement("span");
    coords.className = "coords";
    coords.textContent = person.position
      ? person.position.map((value, axis) => `${"XYZ"[axis]} ${signed(value, 2)}`).join("  ·  ") + " m"
      : `${Math.round(person.score * 100)}% confidence · Range unavailable`;
    li.append(name, direction, coords);
    return li;
  });
  byId("people").replaceChildren(...rows);
}

function connection(label, state) {
  byId("connection-label").textContent = label;
  byId("connection").dataset.state = state;
}

function waiting(message) {
  frame = null;
  image = null;
  depthImage = null;
  connection(message, "waiting");
  videoContext.clearRect(0, 0, video.width, video.height);
  byId("placeholder").hidden = false;
  byId("placeholder-label").textContent = message;
  byId("source").textContent = "Waiting for camera";
  byId("fps").textContent = "";
  byId("resolution").textContent = "";
  byId("count").textContent = "—";
  byId("people").replaceChildren();
  byId("empty").hidden = false;
  byId("empty").textContent = "Waiting for camera.";
  byId("pause").disabled = true;
  byId("overlay").disabled = true;
  cameraViewControls();
  drawPlot();
}

async function poll() {
  try {
    if (paused) return;
    const response = await fetch("/api/frame", {
      cache: "no-store",
      signal: AbortSignal.timeout(4000),
    });
    if (!response.ok) throw new Error("Viewer unavailable");
    const next = await response.json();
    if (next.mode === "waiting") {
      if (frame || byId("placeholder-label").textContent !== "Waiting for camera") waiting("Waiting for camera");
      return;
    }
    if (frame?.sequence === next.sequence) return;
    const nextImage = new Image();
    nextImage.src = `data:image/jpeg;base64,${next.image}`;
    const nextDepth = next.depth_image ? new Image() : null;
    if (nextDepth) nextDepth.src = `data:image/png;base64,${next.depth_image}`;
    await Promise.all([nextImage.decode(), nextDepth?.decode()]);
    if (paused) return;
    frame = next;
    image = nextImage;
    depthImage = nextDepth;
    const metric = next.mode === "rgbd";
    connection(metric ? "Live RGB-D" : "Live RGB", "live");
    byId("placeholder").hidden = true;
    byId("resolution").textContent = `${next.width} × ${next.height} · `;
    byId("source").textContent = metric ? "RealSense RGB + depth" : "RealSense RGB via RDP";
    byId("fps").textContent = next.fps ? ` · ${Math.round(next.fps)} fps` : "";
    byId("pause").disabled = false;
    byId("overlay").disabled = false;
    cameraViewControls();
    drawVideo();
    drawPlot();
    renderDetections(next.people);
  } catch {
    if (!paused) waiting("Reconnecting");
  } finally {
    window.setTimeout(poll, paused ? 350 : 95);
  }
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setCameraMode(button.dataset.cameraMode));
});

window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "y" && !event.ctrlKey && !event.altKey && !event.metaKey
      && !event.repeat && !setupDialog.open && !event.target.matches("input, textarea") && depthImage) {
    setCameraMode(cameraMode === "depth" ? colorMode : "depth");
  }
});

byId("plot-space").addEventListener("click", () => {
  plotMode = "space";
  drawPlot();
});

byId("plot-top").addEventListener("click", () => {
  plotMode = "top";
  drawPlot();
});

function rotatePlot(dx, dy) {
  orbit.yaw += dx;
  orbit.pitch = Math.max(0.12, Math.min(1.42, orbit.pitch + dy));
  drawPlot();
}

byId("reset-orbit").addEventListener("click", () => {
  orbit = { ...defaultOrbit };
  drawPlot();
});

plot.addEventListener("pointerdown", (event) => {
  if (plot.dataset.rotatable !== "true" || event.button !== 0) return;
  drag = [event.clientX, event.clientY];
  plot.setPointerCapture(event.pointerId);
});

plot.addEventListener("pointermove", (event) => {
  if (!drag || plot.dataset.rotatable !== "true") return;
  rotatePlot((event.clientX - drag[0]) * 0.013, (event.clientY - drag[1]) * 0.012);
  drag = [event.clientX, event.clientY];
});

plot.addEventListener("lostpointercapture", () => { drag = null; });

plot.addEventListener("keydown", (event) => {
  if (plot.dataset.rotatable !== "true") return;
  const shifts = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] };
  if (shifts[event.key]) {
    event.preventDefault();
    rotatePlot(...shifts[event.key]);
  }
});

byId("setup-open").addEventListener("click", () => {
  setupDialog.showModal();
});

byId("overlay").addEventListener("click", () => {
  overlays = !overlays;
  byId("overlay").setAttribute("aria-pressed", String(overlays));
  drawVideo();
});

byId("pause").addEventListener("click", () => {
  paused = !paused;
  byId("pause").textContent = paused ? "Resume" : "Pause";
  connection(paused ? "Paused" : "Connecting", "waiting");
  if (!paused) frame = null;
});

if (location.hash) history.replaceState(null, "", location.pathname + location.search);
waiting("Connecting to camera");
poll();
