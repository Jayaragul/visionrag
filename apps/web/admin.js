/* Vision-RAG admin dashboard.
 *
 * Read-mostly. Polls a handful of aggregate endpoints and renders them; the
 * only writes are naming a place and deleting an enrolled person.
 *
 * Polling rather than a websocket: this is an operator view refreshed every
 * few seconds, and it must never compete with the capture path for CPU.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 4000;

let selectedPlace = null;

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function bytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function ago(epochSeconds) {
  if (!epochSeconds) return "—";
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function bar(value, good = 0.7, poor = 0.4) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const cls = value >= good ? "ok" : value >= poor ? "warn" : "bad";
  return `<div class="bar ${cls}" title="${pct}%"><i style="width:${pct}%"></i></div>`;
}

function table(headers, rows, emptyMessage) {
  if (!rows.length) return `<div class="empty">${esc(emptyMessage)}</div>`;
  const head = headers.map((h) =>
    `<th${h.num ? ' class="num"' : ""}>${esc(h.label)}</th>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

/* ---------- sections ---------- */

async function renderOverview() {
  const o = await get("/api/admin/overview");
  const tiles = [
    ["devices connected", o.devices_connected],
    ["places mapped", o.places_mapped],
    ["objects remembered", o.objects_remembered],
    ["visits recorded", o.visits_recorded],
    ["events recorded", o.events_recorded],
    ["people enrolled", o.people_enrolled],
  ];
  $("tiles").innerHTML = tiles
    .map(([label, value]) => `<div class="tile"><b>${value}</b><span>${label}</span></div>`)
    .join("");

  const badge = $("devbadge");
  badge.textContent = `${o.devices_connected} device${o.devices_connected === 1 ? "" : "s"}`;
  badge.className = "badge" + (o.devices_connected ? " live" : "");

  const world = $("worldstate");
  world.textContent = o.world_memory_active ? "world memory active" : "no world memory yet";
  world.className = "badge" + (o.world_memory_active ? " live" : "");
}

async function renderDevices() {
  const d = await get("/api/admin/devices");
  $("devcount").textContent = d.count ? `${d.count} total` : "";
  const rows = d.devices.map((v) => {
    const q = v.quality;
    // Quality is shown because a device streaming unusable frames looks
    // identical to a healthy one on frame counts alone.
    const quality = q
      ? `${bar(q.score)}<span class="pill ${q.usable ? "ok" : "bad"}">${
          q.usable ? "usable" : esc(q.reasons.join(", "))}</span>`
      : "—";
    return `<tr>
      <td><span class="pill ${v.connected ? "ok" : ""}">${v.connected ? "live" : "closed"}</span></td>
      <td><code>${esc(v.session_id)}</code></td>
      <td style="max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
          title="${esc(v.device)}">${esc(v.device)}</td>
      <td class="num">${v.elapsed_s ?? "—"}s</td>
      <td class="num">${v.frames_analysed ?? 0}/${v.frames_received ?? 0}</td>
      <td class="num">${v.latency_p95_ms ?? "—"}</td>
      <td>${v.place_id ? "place " + v.place_id : "—"}</td>
      <td style="min-width:150px">${quality}</td>
    </tr>`;
  });
  $("devices").innerHTML = table(
    [{ label: "" }, { label: "session" }, { label: "device" },
     { label: "uptime", num: true }, { label: "analysed/recv", num: true },
     { label: "p95 ms", num: true }, { label: "place" }, { label: "frame quality" }],
    rows,
    "No devices have connected. Open the capture page on a phone to begin.",
  );
}

async function renderPlaces() {
  const p = await get("/api/admin/places");
  $("placecount").textContent = p.count ? `${p.count} total` : "";
  const rows = p.places.map((pl) => `
    <tr class="clickable" data-place="${pl.place_id}">
      <td><strong>${esc(pl.label || "place " + pl.place_id)}</strong></td>
      <td class="num">${pl.n_visits}</td>
      <td class="num">${pl.n_objects}</td>
      <td style="min-width:90px">${
        pl.avg_coverage != null ? bar(pl.avg_coverage) : "—"}</td>
      <td>${pl.has_gps ? '<span class="pill">gps</span>' : "—"}</td>
      <td>${esc(ago(pl.last_seen))}</td>
      <td><button data-rename="${pl.place_id}">name</button></td>
    </tr>`);
  $("places").innerHTML = table(
    [{ label: "place" }, { label: "visits", num: true }, { label: "objects", num: true },
     { label: "avg coverage" }, { label: "" }, { label: "last seen" }, { label: "" }],
    rows,
    p.note || "Nothing mapped yet.",
  );

  $("places").querySelectorAll("tr.clickable").forEach((tr) => {
    tr.onclick = (e) => {
      if (e.target.dataset.rename) return;
      selectPlace(Number(tr.dataset.place));
    };
  });
  $("places").querySelectorAll("[data-rename]").forEach((b) => {
    b.onclick = async (e) => {
      e.stopPropagation();
      const id = Number(b.dataset.rename);
      const label = prompt("Name this place (e.g. Kitchen, Desk):");
      if (!label) return;
      await fetch(`/api/admin/places/${id}/label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      renderPlaces();
    };
  });
}

async function selectPlace(placeId) {
  selectedPlace = placeId;
  await renderObjects();
}

async function renderObjects() {
  if (selectedPlace == null) return;
  const section = $("objsection");
  let data;
  try {
    data = await get(`/api/admin/places/${selectedPlace}/objects`);
  } catch {
    section.style.display = "none";
    return;
  }
  section.style.display = "";
  $("objplace").textContent = `place ${selectedPlace}`;

  const rows = data.objects.map((o) => {
    const kindClass = (o.semantic_kind || "").replace("_object", "");
    const hit = o.opportunities ? o.times_seen / o.opportunities : 0;
    return `<tr>
      <td><strong>${esc(o.class)}</strong></td>
      <td><span class="pill ${esc(kindClass)}">${esc(o.semantic_kind)}</span></td>
      <td><span class="pill">${esc(o.tier)}</span></td>
      <td>${o.state === "tentative" ? '<span class="pill warn">tentative</span>' : ""}</td>
      <td style="min-width:90px">${bar(o.persistence, 0.6, 0.25)}</td>
      <td class="num">${o.times_seen}/${o.opportunities}</td>
      <td style="min-width:70px">${bar(hit, 0.7, 0.4)}</td>
      <td class="num">${o.age_days}d</td>
      <td>${esc(ago(o.last_seen))}</td>
    </tr>`;
  });
  $("objects").innerHTML = table(
    [{ label: "object" }, { label: "kind" }, { label: "tier" }, { label: "" },
     { label: "still there" }, { label: "seen", num: true }, { label: "hit rate" },
     { label: "age", num: true }, { label: "last seen" }],
    rows,
    "Nothing remembered at this place yet.",
  );
}

async function renderEvents() {
  const e = await get("/api/admin/events");
  $("evcount").textContent = Object.keys(e.by_type).length
    ? Object.entries(e.by_type).map(([t, n]) => `${t} ${n}`).slice(0, 3).join(" · ")
    : "";
  const rows = e.events.slice(0, 14).map((ev) => `
    <tr>
      <td class="num">${ev.t_start_s}s</td>
      <td>${esc(ev.type.replace(/_/g, " "))}</td>
      <td class="num">${ev.confidence}</td>
      <td>${ev.ego_suspect ? '<span class="pill warn">camera?</span>' : ""}</td>
    </tr>`);
  $("events").innerHTML = table(
    [{ label: "t", num: true }, { label: "event" },
     { label: "conf", num: true }, { label: "" }],
    rows,
    "No events recorded yet.",
  );
}

async function renderStorage() {
  const s = await get("/api/admin/storage");
  const dbRows = Object.entries(s.databases).map(([name, size]) => `
    <tr><td>${esc(name)}</td><td class="num">${bytes(size)}</td></tr>`);
  dbRows.push(`<tr><td>evidence frames (${s.evidence_files})</td>
                   <td class="num">${bytes(s.evidence_bytes)}</td></tr>`);
  dbRows.push(`<tr><td><strong>total</strong></td>
                   <td class="num"><strong>${bytes(s.total_bytes)}</strong></td></tr>`);
  $("storage").innerHTML = table(
    [{ label: "store" }, { label: "size", num: true }], dbRows, "Nothing stored yet.");
}

async function renderPeople() {
  const p = await get("/api/admin/people");
  $("peoplecount").textContent = p.count ? `${p.count}` : "";
  const rows = p.people.map((person) => `
    <tr>
      <td><strong>${esc(person.name)}</strong></td>
      <td class="num">${person.n_templates}</td>
      <td style="color:var(--faint);font-size:12px">${esc(person.consent_note || "no note")}</td>
      <td><button class="danger" data-forget="${person.person_id}">forget</button></td>
    </tr>`);
  $("people").innerHTML = table(
    [{ label: "person" }, { label: "templates", num: true },
     { label: "consent" }, { label: "" }],
    rows,
    p.note || "Nobody enrolled. Face recognition is off until someone is.",
  );
  $("people").querySelectorAll("[data-forget]").forEach((b) => {
    b.onclick = async () => {
      const id = Number(b.dataset.forget);
      if (!confirm("Delete this person and every biometric template of them?\nThis cannot be undone.")) return;
      await fetch(`/api/admin/people/${id}`, { method: "DELETE" });
      renderPeople();
    };
  });
}

/* ---------- loop ---------- */

async function refresh() {
  try {
    await Promise.all([
      renderOverview(), renderDevices(), renderPlaces(),
      renderEvents(), renderStorage(), renderPeople(), renderObjects(),
    ]);
    $("refreshed").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (err) {
    $("refreshed").textContent = "error: " + err.message;
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
