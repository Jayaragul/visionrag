# visionrag

**CPU-first spatial memory for a phone camera.** Point a phone at a room; the
system detects and tracks objects, builds a typed event timeline, learns which
things are permanent and which are passing through, and answers questions with
cited evidence — with no GPU, and no VLM in the ingest loop.

It answers two questions:

- **What's around me?** — what is here, and which of it is fixtures, furniture,
  or people.
- **What's different?** — what changed since the last time I was here.

Everything runs locally. Nothing leaves the machine.

---

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Jayaragul/visionrag.git
```

```bash
pip install -r requirements.txt
```

Run the pipeline end-to-end on a generated clip — no model download, no
network, no camera:

```bash
PYTHONPATH=src python -m visionrag demo
```

Expected output:

```
0.00s  appeared        tracks=[1]
0.00s  appeared        tracks=[2]
0.20s  approached      tracks=[1, 2]
4.00s  near            tracks=[1, 2]
4.00s  remained_near   tracks=[1, 2]
4.40s  stopped         tracks=[1]
8.00s  moved_away      tracks=[1, 2]
8.60s  started_moving  tracks=[1]
```

See the persistent world model learn what belongs in a place, over a simulated
week of visits:

```bash
python scripts/demo_world.py
```

```
object         kind            tier           still there    seen
dining table   furniture       static               1.000     7/7
tv             furniture       static               0.999     6/7   <- survived a missed detection
laptop         movable_object  semi_static          0.006     3/7   <- correctly flagged removed
person         person          dynamic              0.000     6/7   <- present 6/7 visits, still not furniture

gone       laptop
unchanged  dining table, tv
```

---

## Running with a phone camera

The phone is a sensor and a display; all inference happens on your machine.

**1. Generate a TLS certificate.** `getUserMedia` only works in a secure
context — `localhost` is exempt, but your phone is not localhost, so a LAN test
needs HTTPS.

```bash
python scripts/make_cert.py
```

**2. Start the server.**

```bash
python apps/api/server.py --cert certs/dev.pem
```

**3. Open the printed URL on a phone on the same Wi-Fi**, e.g.
`https://192.168.1.6:8443`. Accept the certificate warning once — it is
self-signed, and the warning is correct. iOS is stricter than Android and may
need the certificate trusted under *Settings › General › VPN & Device
Management*.

Tap **Start camera**. You should see live boxes, a filling event timeline, and
a question box.

The phone previews at its native frame rate locally and uploads only sampled
JPEG frames. Exactly one frame is in flight at a time — without that
backpressure the queue grows without bound the moment the server falls behind,
and latency never recovers.

### Recorded video instead

```bash
PYTHONPATH=src python -m visionrag ingest clip.mp4 --backend torchvision --fps 2
```

```bash
PYTHONPATH=src python -m visionrag timeline --db runs/memory.db
```

---

## How it works

```
phone camera ─> detect ─> track ─> ego-motion ─> events ─> event memory
                            │                                  │
                            │                          "what just happened?"
                            ▼
                     place recognition
                            │
                   place id + live→canonical transform
                            │
                            ▼
                       world memory
                    ┌───────┴────────┐
              current visit    previous visits
                    └───────┬────────┘
                            ▼
              "what's around?"   "what's different?"
```

### Four orthogonal axes

Conflating these is how a person who sits still gets filed as building
structure. Semantic kind comes from the detector class and **never** from
motion:

```
class            person | chair | desk | laptop | backpack
semantic_kind    person | furniture | fixture | movable_object
persistence_tier static | semi_static | dynamic
motion_state     moving | stationary | unknown
```

Kind constrains which tiers are reachable: a person is never `static` however
reliably observed, and a backpack is never a fixture.

### Absence requires coverage

"I looked and it's gone" and "I never looked over there" are different claims.

| Status | Meaning | Evidence of removal? |
|---|---|---|
| `seen` | observed and detected | — |
| `checked_missing` | expected area in view, unoccluded, absent | **yes** |
| `occluded_uncertain` | in view but something was blocking it | no |
| `not_observed` | never looked | no |

Only `checked_missing` reaches the persistence filter. The others append
nothing, so belief decays under the survival prior alone — exactly what "I
haven't checked in three days" should mean. This is the object-level form of
the occupancy-grid distinction between *observed empty* and *unobserved*.

### Persistence

Whether something is still there is estimated from detections **and misses**,
using the persistence filter of Rosen, Mason & Leonard (ICRA 2016): an
exponential survival prior per object class, updated by observations that
account for detector fallibility. Six consecutive absences drive belief below
0.05; a single miss among many sightings keeps it above 0.80.

---

## Measured performance

Re-run on any new host — **these numbers do not transfer**:

```bash
python scripts/bench_detector.py --json runs/bench.json
```

Intel 12th-gen mobile, 10 physical / 16 logical cores, no GPU,
`ssdlite320_mobilenet_v3_large` via torchvision eager:

| Input | Threads | Wall ms | CPU ms | Ceiling |
|------:|--------:|--------:|-------:|--------:|
| 320 | 1 | 234.7 | 218.8 | 4.3 fps |
| 320 | 8 | 186.0 | 540.6 | 5.4 fps |
| 640 | 8 | 214.5 | 626.0 | 4.7 fps |

Threading scales badly — 2.5× the CPU for 1.26× the speed. Eager PyTorch is the
bottleneck, not the model: SSDLite is ~0.6 GFLOPs and should be several times
faster. **ONNX export is the obvious next optimisation**; these are the numbers
it has to beat. Live config is capped at 3 fps, below the ~4.9 fps end-to-end
ceiling, leaving slack for JPEG decode, network jitter and the OS.

### Ego-motion compensation

Handheld video makes the whole scene translate, and a naive event layer reads
that as every object moving. Measured on a fixture where true object motion is
unchanged, so every extra event is a false positive:

| Condition | Events | False motion events |
|---|---:|---:|
| Static camera (ground truth) | 8 | — |
| Shake, compensation **off** | 20 | **12** |
| Shake, compensation **on** | 9 | **0** |

```bash
PYTHONPATH=src python -m visionrag demo --shake --no-ego
```

### Place recognition: retrieve, then verify

Panning the camera across a synthetic panorama:

| Pan | Appearance score | ORB inliers | Warp usable |
|---|---|---|---|
| 5% | 0.927 | 168 | yes |
| **15%** | **0.734** ✗ | 137 | yes |
| 40% | 0.508 ✗ | 114 | yes |

The tiled-histogram descriptor falls below threshold at **~12% pan**, while ORB
still verifies the same place at 40%. Gating on appearance meant the stronger
signal never got consulted — one place fragmented into three across a 50% pan.
Appearance is now a top-5 shortlist and geometry decides; the same sweep now
yields one place.

**Degenerate homographies pass every obvious check.** One fit had 37 inliers and
0.005 reprojection error while collapsing the whole frame to a single point.
Mapped-quad area, convexity and reflection are therefore checked explicitly.

---

## Where data is stored

**Nothing on the phone** beyond the live preview in browser memory — close the
tab and it is gone. **Nothing leaves the machine**: the query layer is local and
rule-based, with no LLM call.

Everything lives under `runs/`:

| Data | Location | Growth |
|---|---|---|
| Runs, frames, observations, tracks, events | `runs/**/memory.db` | ~6 MB/hr |
| Evidence frames | `runs/**/evidence/*.jpg` | ~25 MB/hr |
| Places, object instances, visit history | `runs/**/world.db` | small |

Retention is set by `store.retention_mode` in `configs/live.yaml`:

- `metadata` — no pixels ever written (~6 MB/hr)
- `evidence` — JPEG kept only for frames supporting an event (~30 MB/hr, default)
- `full` — every analysed frame (~5 GB/hr; short debug clips only)

`DELETE /api/sessions/{id}` removes the rows and the evidence files. GPS is off
by default and is never required.

> Growth figures are calculated, not yet measured against a long real session.

---

## Layout

```
src/visionrag/
  types.py          core contracts: Detection, Track, Event, EventType
  cost.py           per-stage CPU/wall accounting
  config.py         one config object per run, hashed for reproducibility
  pipeline.py       orchestration; shared by file and live ingest
  ingest/           video, detect, track, egomotion, scheduler
  memory/
    events.py       event induction rules
    store.py        per-run SQLite store
    persistence.py  persistence filter, semantic kinds, tiers
    places.py       place recognition, homography, coverage geometry
    world.py        cross-visit object instances and change detection
apps/api/           FastAPI gateway, live session, grounded query
apps/web/           phone client (no build step, no framework, no CDN)
scripts/            benchmark, cert generation, world demo
tests/              pytest; stub detector, so no network needed
```

## Development

```bash
python -m pytest tests/ -q
```

43 tests, all offline. They use a colour-threshold stub detector so CI needs no
checkpoint and no network.

---

## Status

**Working:** live phone capture, detection, tracking, ego-motion compensation,
typed events, grounded query with abstention, persistent place recognition,
object persistence with semantic kinds, observation coverage, occlusion
handling, object lifecycle, change detection.

**Not done:** `MOVED` as a distinct change type (needs appearance re-identification),
visit-to-visit diffing, the "Around Me" phone panel, GPS in the client, ONNX
export.

**The main caveat:** every measurement above comes from synthetic scenes.
Behaviour under real viewpoint and lighting change is **unvalidated**, and the
appearance descriptor is the fragile part. Validating place recognition on real
photographs is the gate before further work.

## References

- Rosen, Mason & Leonard, *Towards Lifelong Feature-Based Mapping in Semi-Static
  Environments* (ICRA 2016) — the persistence filter
- Moravec & Elfes, occupancy grids (1985) — unknown is not empty
- Gálvez-López & Tardós, *Bags of Binary Words* (T-RO 2012) — retrieve-then-verify
- Hughes, Chang & Carlone, *Hydra* (RSS 2022) — layered spatial world models

*Citations are from memory and should be verified before use in publication.*
