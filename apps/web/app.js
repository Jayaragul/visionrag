/* Phone client.
 *
 * The phone is a sensor and a display. It previews at the camera's native rate
 * locally (free) and uploads only sampled JPEG frames.
 *
 * Backpressure: exactly one frame is in flight at any time. The next frame is
 * not captured until the previous result returns. Without this the send queue
 * grows without bound the instant the server falls behind, and the overlay
 * drifts further from reality the longer the session runs.
 *
 * What the UI shows is deliberately ordered: where you are, what changed,
 * what's here — then, only if asked, frame rates and latency. Those are
 * debugging tools and they used to be the first thing on screen.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");

let session = null;
let ws = null;
let stream = null;
let running = false;
let inFlight = false;       // the backpressure gate
let lastSendAt = 0;
let placePollTimer = null;

const seenEvents = new Set();
const trackClasses = new Map();   // track_id -> class, for readable phrasing
let eventCount = 0;
const latencies = [];
const frameTimes = [];

const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { willReadFrequently: true });

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const log = (m) => { $("log").textContent = m || ""; };

/* ---------- access token ----------
 * The server prints a link containing ?t=<token>. It is stashed in
 * sessionStorage so a reload or an in-app navigation does not lose access,
 * and stripped from the address bar so it does not end up in a screenshot or
 * a shared URL. sessionStorage (not localStorage) means it dies with the tab.
 */
const TOKEN = (() => {
  const fromUrl = new URLSearchParams(location.search).get("t");
  if (fromUrl) {
    sessionStorage.setItem("vr_token", fromUrl);
    history.replaceState(null, "", location.pathname);
    return fromUrl;
  }
  return sessionStorage.getItem("vr_token") || "";
})();

function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
  return fetch(path, { ...opts, headers });
}

// A browser cannot set headers on a WebSocket handshake, so the token rides
// as a query parameter there.
const wsUrl = (path) =>
  `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}` +
  (TOKEN ? `?t=${encodeURIComponent(TOKEN)}` : "");

/* ---------- readable event phrasing ----------
 * Mirrors apps/api/phrasing.py. Duplicated rather than fetched because the
 * timeline updates per frame and a round trip per line would be absurd; the
 * two must be kept in step. */
const ARTICLE = {
  person: "someone", cat: "a cat", dog: "a dog", car: "a car",
  bicycle: "a bicycle", chair: "a chair", laptop: "a laptop", cup: "a cup",
  bottle: "a bottle", backpack: "a backpack", handbag: "a bag",
  book: "a book", "cell phone": "a phone", tv: "the screen",
  "dining table": "the table", couch: "the couch", bed: "the bed",
  refrigerator: "the fridge", "potted plant": "a plant",
  object_a: "the first object", object_b: "the second object",
};
const noun = (c) => ARTICLE[c] || String(c || "something").replace(/_/g, " ");

const TEMPLATES = {
  appeared: "{s} came into view", reappeared: "{s} came back",
  disappeared: "{s} went out of view", started_moving: "{s} started moving",
  stopped: "{s} stopped", direction_changed: "{s} changed direction",
  entered_region: "{s} entered the area", left_region: "{s} left the area",
  near: "{s} came close together", approached: "{s} moved closer together",
  moved_away: "{s} moved apart", remained_near: "{s} stayed close together",
  camera_moved: "the camera started moving", camera_stabilised: "the camera settled",
};

function subject(ids) {
  const names = ids.map((i) => noun(trackClasses.get(i))).filter(Boolean);
  if (!names.length) return "something";
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return names.slice(0, -1).join(", ") + ` and ${names[names.length - 1]}`;
}

function describeEvent(ev) {
  const t = TEMPLATES[ev.type];
  if (!t) return ev.type.replace(/_/g, " ");
  const s = t.replace("{s}", subject(ev.participants || []));
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/* ---------- camera ---------- */

async function startCamera() {
  $("start").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" },
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    $("start").disabled = false;
    // Almost always a non-secure context rather than a denied permission.
    log(window.isSecureContext
      ? `Camera unavailable: ${err.message}`
      : "This page needs https:// for camera access. Start the server with --cert.");
    return;
  }

  video.srcObject = stream;
  await video.play();
  $("stage").classList.remove("idle");
  $("idlecopy").style.display = "none";
  video.style.display = overlay.style.display = "";
  setStatus("live", "Finding this place…", "Comparing what you see with places already known");

  const res = await api("/api/sessions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ retention_mode: "evidence" }),
  });
  if (res.status === 401) {
    $("start").disabled = false;
    log("This link is missing its access token. Open the link the server printed.");
    return;
  }
  session = await res.json();
  $("deletewrap").style.display = "";

  ws = new WebSocket(wsUrl(`/api/sessions/${session.session_id}/stream`));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    running = true;
    $("stop").disabled = false;
    pump();
    // The place is resolved server-side after the first analysed frame, so
    // poll rather than waiting for a frame result to carry it.
    placePollTimer = setInterval(refreshPlace, 2500);
  };
  ws.onmessage = (e) => { inFlight = false; onResult(JSON.parse(e.data)); };
  ws.onclose = () => { running = false; };
  ws.onerror = () => log("Connection problem.");
}

function captureJpeg() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  // Downscale on the phone: uploading 1280px frames wastes bandwidth on
  // pixels the server immediately discards.
  const scale = Math.min(1, 480 / Math.max(vw, vh));
  grab.width = Math.round(vw * scale);
  grab.height = Math.round(vh * scale);
  grabCtx.drawImage(video, 0, 0, grab.width, grab.height);
  return new Promise((r) => grab.toBlob(r, "image/jpeg", 0.7));
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
  if (msg.error) { log("Error: " + msg.error); return; }
  if (msg.skipped) return;

  latencies.push(msg.latency_ms);
  if (latencies.length > 20) latencies.shift();

  for (const d of msg.detections || []) trackClasses.set(d.track_id, d["class"]);
  draw(msg.detections || []);

  $("s-lat").textContent = Math.round(
    latencies.reduce((a, b) => a + b, 0) / latencies.length);
  $("s-obj").textContent = (msg.detections || []).length;
  if (frameTimes.length > 1) {
    const span = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
    $("s-fps").textContent = span > 0
      ? ((frameTimes.length - 1) / span).toFixed(1) : "—";
  }

  showHint(msg.quality, msg.place);
  (msg.events || []).forEach(addEvent);
}

/* Actionable guidance, not a diagnostic readout: a person can act on
   "hold steady", not on "sharpness 18.4". */
const QUALITY_COPY = {
  "too dark": "Too dark to see clearly — try more light",
  "overexposed": "Too bright — try pointing away from the light",
  "no contrast (lens covered?)": "The camera may be covered",
  "blurred": "Hold steady — the view is blurred",
  "heavily clipped": "Harsh lighting is washing out the view",
};

function showHint(quality, place) {
  const existing = document.querySelector(".hint");
  let text = null;
  if (quality && !quality.usable) {
    text = (quality.reasons || []).map((r) => QUALITY_COPY[r]).find(Boolean)
        || "Camera view is unclear";
  } else if (place && place.rejected) {
    text = "Not recording — the view isn't clear enough";
  }
  if (!text) { if (existing) existing.remove(); return; }
  if (existing) { existing.lastChild.textContent = text; return; }
  const el = document.createElement("div");
  el.className = "hint";
  el.appendChild(document.createTextNode(text));
  $("stage").appendChild(el);
}

function draw(dets) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w) return;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);

  const scale = w / 400;
  ctx.lineWidth = Math.max(2, 2.2 * scale);
  ctx.font = `600 ${Math.max(12, 12 * scale)}px -apple-system, sans-serif`;
  ctx.textBaseline = "middle";

  for (const d of dets) {
    const x = d.box.x * w, y = d.box.y * h;
    const bw = d.box.w * w, bh = d.box.h * h;
    // Rounded, semi-transparent boxes read as an interface rather than a
    // debug dump, and stay legible over a moving scene.
    ctx.strokeStyle = "rgba(90,169,255,.95)";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, bw, bh, 6 * scale);
    else ctx.rect(x, y, bw, bh);
    ctx.stroke();

    const label = noun(d["class"]);
    const padX = 7 * scale, tw = ctx.measureText(label).width + padX * 2;
    const th = 20 * scale;
    const ly = Math.max(th / 2 + 2, y - th / 2 - 3 * scale);
    ctx.fillStyle = "rgba(90,169,255,.95)";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, ly - th / 2, tw, th, 5 * scale);
    else ctx.rect(x, ly - th / 2, tw, th);
    ctx.fill();
    ctx.fillStyle = "#04121f";
    ctx.fillText(label, x + padX, ly);
  }
}

/* ---------- place ---------- */

function setStatus(cls, name, sub) {
  $("status").className = "status " + cls;
  const el = $("placename");
  el.textContent = name;
  el.classList.toggle("unnamed", cls !== "known");
  $("placesub").textContent = sub;
}

async function refreshPlace() {
  if (!session) return;
  let p;
  try { p = await (await api(`/api/sessions/${session.session_id}/place`)).json(); }
  catch { return; }

  if (!p.known) {
    setStatus("locating", "Finding this place…",
              "Comparing what you see with places already known");
    return;
  }

  $("namebtn").style.display = "";
  setStatus(p.named ? "known" : "live",
            p.title,
            p.first_visit ? "First time here — I'll remember it" : p.subtitle);

  // Changes first: this is the reason the product exists, and burying it
  // under an inventory list would waste the one thing nothing else does.
  const csec = $("changesec");
  if (p.first_visit) {
    csec.style.display = "";
    $("changes").innerHTML =
      `<div class="empty">Nothing to compare yet. Come back and I'll tell you
       what moved, what's gone, and what's new.</div>`;
  } else {
    csec.style.display = "";
    const head = `<div class="change-head${p.changes.length ? "" : " quiet"}">
                    ${esc(p.changes_summary)}</div>`;
    const rows = p.changes.map((c) => `
      <div class="row ${c.direction === "removed" ? "gone" : "added"}">
        <span class="dot"></span><span class="text">${esc(c.text)}</span>
      </div>`).join("");
    $("changes").innerHTML = head + rows;
  }

  const hsec = $("heresec");
  if (p.here.length) {
    hsec.style.display = "";
    $("herecount").textContent = p.here.length;
    $("here").innerHTML = p.here.map((o) => `
      <div class="row">
        <span class="dot ${esc(o.tone)}"></span>
        <span>${esc(o.name)}</span>
        <span class="meta">${esc(o.belief)}<br>
          <span style="opacity:.7">${esc(o.kind)}</span></span>
      </div>`).join("");
  } else {
    hsec.style.display = "";
    $("here").innerHTML =
      `<div class="empty">Still watching. Objects appear here once they've been
       seen enough to be worth remembering.</div>`;
  }
}

async function namePlace() {
  if (!session) return;
  const label = prompt("What is this place called?\n\ne.g. Kitchen, My desk, Garage");
  if (!label) return;
  await api(`/api/sessions/${session.session_id}/place/label`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  refreshPlace();
}

/* ---------- activity ---------- */

function addEvent(ev) {
  const key = `${ev.type}:${ev.t_start_ms}:${(ev.participants || []).join(",")}`;
  if (seenEvents.has(key)) return;
  seenEvents.add(key);
  eventCount += 1;
  $("evcount").textContent = eventCount;

  const row = document.createElement("div");
  row.className = "ev";
  row.innerHTML =
    `<time>${(ev.t_start_ms / 1000).toFixed(0)}s</time>` +
    `<span>${esc(describeEvent(ev))}</span>` +
    (ev.ego_suspect ? `<span class="qual">may be camera</span>` : "");
  const tl = $("timeline");
  tl.prepend(row);
  while (tl.children.length > 80) tl.lastChild.remove();
}

/* ---------- ask ---------- */

async function ask() {
  const question = $("q").value.trim();
  if (!question) return;
  if (!session) { log("Start the camera first."); return; }
  $("askbtn").disabled = true;
  try {
    const a = await (await api(`/api/sessions/${session.session_id}/query`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })).json();
    const box = $("answer");
    box.style.display = "block";
    box.classList.toggle("abstain", a.abstained);
    // Citations are shown as a plain count and time span. Saying "0.82
    // confidence" invites a precision the estimate does not have.
    const cite = a.abstained
      ? "Nothing recorded here supports an answer yet."
      : `Based on ${a.evidence_ids.length} recorded observation${
          a.evidence_ids.length === 1 ? "" : "s"}, ${a.time_range[0]}s–${a.time_range[1]}s`;
    box.innerHTML = `<div class="text">${esc(a.answer)}</div>
                     <div class="cite">${esc(cite)}</div>`;
  } finally {
    $("askbtn").disabled = false;
  }
}

/* ---------- teardown ---------- */

async function stopCamera() {
  running = false;
  clearInterval(placePollTimer);
  $("stop").disabled = true;
  $("start").disabled = false;
  $("start").textContent = "Start camera";
  if (ws && ws.readyState === 1) ws.close();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.style.display = overlay.style.display = "none";
  $("stage").classList.add("idle");
  $("idlecopy").style.display = "";
  document.querySelector(".hint")?.remove();
  $("status").className = "status";
  if (session) {
    await api(`/api/sessions/${session.session_id}/stop`, { method: "POST" });
    $("placesub").textContent = "Saved. Come back and I'll tell you what changed.";
  }
}

async function deleteSession(e) {
  e.preventDefault();
  if (!session) return;
  if (!confirm("Delete everything recorded in this session?\nThis cannot be undone."))
    return;
  await stopCamera();
  await api(`/api/sessions/${session.session_id}`, { method: "DELETE" });
  ["changes", "here", "timeline"].forEach((id) => ($(id).innerHTML = ""));
  $("changesec").style.display = $("heresec").style.display = "none";
  $("answer").style.display = "none";
  $("deletewrap").style.display = "none";
  session = null;
  eventCount = 0;
  seenEvents.clear();
  setStatus("", "Not started", "Point your camera at a room to begin");
  log("Session deleted.");
}

$("start").onclick = startCamera;
$("stop").onclick = stopCamera;
$("askbtn").onclick = ask;
$("namebtn").onclick = namePlace;
$("deletelink").onclick = deleteSession;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });

if (!window.isSecureContext) {
  log("This page needs https:// for camera access — see scripts/make_cert.py");
}
