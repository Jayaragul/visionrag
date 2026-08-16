/* Vision-RAG live client.
 *
 * The phone is a sensor and a display. It previews at the camera's native
 * rate locally (free) and sends only sampled JPEG frames to the server.
 *
 * The important detail is backpressure: exactly one frame is in flight at any
 * time. The next frame is not captured until the previous result comes back.
 * Without this the send queue grows without bound the instant the server falls
 * behind, and latency never recovers -- the overlay would drift further from
 * reality the longer the session ran.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");

let session = null;      // {session_id, target_fps, ...}
let ws = null;
let stream = null;
let running = false;
let inFlight = false;    // the backpressure gate
let lastSendAt = 0;
let eventCount = 0;
let seenEvents = new Set();
let latencies = [];
let frameTimes = [];

const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { willReadFrequently: true });

function log(msg) {
  $("log").textContent = msg;
}

function setCameraState(live, text) {
  $("dot").classList.toggle("live", live);
  $("camtext").textContent = text;
}

/* ---------- capture ---------- */

async function startCamera() {
  $("start").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" },
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,   // audio is off by default for this product (PRD 9)
    });
  } catch (err) {
    $("start").disabled = false;
    // The overwhelmingly common cause is a non-secure context.
    const insecure = !window.isSecureContext;
    log("Camera failed: " + err.name + (insecure
      ? " — page is not a secure context. Phone browsers only allow camera "
        + "access over https:// (or localhost). Start the server with --cert."
      : " — " + err.message));
    return;
  }

  video.srcObject = stream;
  await video.play();
  setCameraState(true, "camera active");

  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ retention_mode: "evidence" }),
  });
  session = await res.json();

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/api/sessions/${session.session_id}/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    running = true;
    $("stop").disabled = false;
    log(`session ${session.session_id} · detector ${session.detector} · `
        + `target ${session.target_fps} fps`);
    pump();
  };
  ws.onmessage = (e) => { inFlight = false; onResult(JSON.parse(e.data)); };
  ws.onclose = () => { running = false; log("stream closed"); };
  ws.onerror = () => log("websocket error");
}

function captureJpeg() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  // Downscale on the phone: sending 1280px frames wastes bandwidth and
  // upload time for pixels the server immediately throws away.
  const long = 480;
  const scale = Math.min(1, long / Math.max(vw, vh));
  grab.width = Math.round(vw * scale);
  grab.height = Math.round(vh * scale);
  grabCtx.drawImage(video, 0, 0, grab.width, grab.height);
  return new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.7));
}

async function pump() {
  if (!running) return;
  const interval = 1000 / (session.target_fps || 2);
  const now = performance.now();

  if (!inFlight && now - lastSendAt >= interval && ws.readyState === 1) {
    const blob = await captureJpeg();
    if (blob) {
      inFlight = true;
      lastSendAt = now;
      frameTimes.push(now);
      if (frameTimes.length > 20) frameTimes.shift();
      ws.send(await blob.arrayBuffer());
    }
  }
  requestAnimationFrame(pump);
}

/* ---------- results ---------- */

function onResult(msg) {
  if (msg.error) { log("server error: " + msg.error); return; }
  if (msg.skipped) return;   // scheduler declined this frame; not an error

  latencies.push(msg.latency_ms);
  if (latencies.length > 20) latencies.shift();

  draw(msg.detections || []);
  $("s-lat").textContent = Math.round(
    latencies.reduce((a, b) => a + b, 0) / latencies.length);
  $("s-obj").textContent = (msg.detections || []).length;

  if (frameTimes.length > 1) {
    const span = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
    $("s-fps").textContent = span > 0
      ? ((frameTimes.length - 1) / span).toFixed(1) : "-";
  }

  if (msg.camera) {
    setCameraState(true, msg.camera.moving ? "camera active · moving"
                                           : "camera active · steady");
  }
  (msg.events || []).forEach(addEvent);
}

function draw(dets) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w) return;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.font = `${Math.max(13, w / 40)}px -apple-system, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of dets) {
    const x = d.box.x * w, y = d.box.y * h;
    const bw = d.box.w * w, bh = d.box.h * h;
    ctx.strokeStyle = "#4ea1ff";
    ctx.strokeRect(x, y, bw, bh);
    const label = `#${d.track_id} ${d["class"]} ${d.confidence.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 10;
    const th = parseInt(ctx.font) + 7;
    ctx.fillStyle = "rgba(78,161,255,.9)";
    ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = "#05131f";
    ctx.fillText(label, x + 5, Math.max(0, y - th) + 3);
  }
}

function addEvent(ev) {
  // Events can repeat across frames while a condition holds; key on identity
  // so the timeline shows each occurrence once.
  const key = `${ev.type}:${ev.t_start_ms}:${ev.participants.join(",")}`;
  if (seenEvents.has(key)) return;
  seenEvents.add(key);
  eventCount += 1;
  $("s-ev").textContent = eventCount;

  const row = document.createElement("div");
  row.className = "ev" + (ev.ego_suspect ? " suspect" : "");
  const who = ev.participants.length
    ? ev.participants.map((p) => "#" + p).join(" + ") : "scene";
  row.innerHTML =
    `<time>${(ev.t_start_ms / 1000).toFixed(1)}s</time>` +
    `<span>${ev.type.replace(/_/g, " ")}</span>` +
    (ev.ego_suspect ? `<span class="flag">camera?</span>` : "") +
    `<span class="who">${who} · ${ev.confidence.toFixed(2)}</span>`;
  const tl = $("timeline");
  tl.prepend(row);
  while (tl.children.length > 120) tl.lastChild.remove();
}

/* ---------- query ---------- */

async function ask() {
  const question = $("q").value.trim();
  if (!question || !session) return;
  $("ask").disabled = true;
  try {
    const res = await fetch(`/api/sessions/${session.session_id}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const a = await res.json();
    const box = $("answer");
    box.style.display = "block";
    box.classList.toggle("abstain", a.abstained);
    let meta = a.abstained
      ? "No supporting evidence stored."
      : `confidence ${a.confidence} · ${a.time_range[0]}s–${a.time_range[1]}s · `
        + `${a.evidence_ids.length} evidence ids`;
    if (a.limitations) meta += ` · ${a.limitations}`;
    box.innerHTML = `<div>${a.answer}</div><div class="meta">${meta}</div>`;
  } finally {
    $("ask").disabled = false;
  }
}

/* ---------- teardown ---------- */

async function stopCamera() {
  running = false;
  $("stop").disabled = true;
  $("start").disabled = false;
  if (ws && ws.readyState === 1) ws.close();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  setCameraState(false, "camera off");
  if (session) {
    const r = await fetch(`/api/sessions/${session.session_id}/stop`,
                          { method: "POST" });
    const s = await r.json();
    if (s.cost) {
      log(`stopped · ${s.cost.total_cpu_s}s CPU used · `
          + `${eventCount} events stored`);
    }
  }
}

async function deleteSession() {
  if (!session) return;
  if (!confirm("Delete all stored data for this session?")) return;
  await stopCamera();
  await fetch(`/api/sessions/${session.session_id}`, { method: "DELETE" });
  $("timeline").innerHTML = "";
  $("answer").style.display = "none";
  eventCount = 0; seenEvents = new Set();
  $("s-ev").textContent = "0";
  session = null;
  log("session deleted");
}

$("start").onclick = startCamera;
$("stop").onclick = stopCamera;
$("delete").onclick = deleteSession;
$("ask").onclick = ask;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });

if (!window.isSecureContext) {
  log("This page is not a secure context. Phone browsers will refuse camera "
      + "access — serve over https:// (see scripts/make_cert.py).");
}
