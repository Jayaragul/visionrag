"""Persistent world memory.

Ties place recognition to object persistence so the system can answer
questions that span visits:

    "Is the chair still there?"
    "What changed since I was last here?"
    "Where did I last see the backpack?"

Three things separate this from the per-run event store:

* **Instances outlive runs.** An object instance is scoped to a *place*, not a
  session, so it accumulates evidence across visits.
* **Absence is recorded** -- but only where absence was actually established.
* **Anchors are canonical.** Object positions are warped into the place's
  reference frame, so standing two steps to the left does not create a second
  copy of every object.

Coverage
--------
The critical rule: *"I looked and it was gone"* and *"I never looked over
there"* are different claims. Glancing at one corner of a room must not mark
everything else removed. This is the object-level form of the distinction
occupancy-grid mapping draws between observed-empty and unobserved (Moravec &
Elfes, 1985) -- unknown is not absent.

Timestamps are absolute epoch seconds, not session-relative milliseconds,
because persistence is reasoned about across days.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..types import Track
from .persistence import (
    InstanceState,
    ObservationStatus,
    PersistenceFilter,
    SemanticKind,
    Tier,
    classify,
    is_evidence,
    lifetime_prior,
    semantic_kind,
)
from .places import GeoFix, Place, PlaceIndex, PlaceMatch

WORLD_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS places (
    place_id    INTEGER PRIMARY KEY,
    descriptor  BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    label       TEXT,
    lat         REAL,
    lon         REAL,
    accuracy_m  REAL,
    n_visits    INTEGER NOT NULL DEFAULT 0,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS place_visits (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id INTEGER NOT NULL REFERENCES places(place_id) ON DELETE CASCADE,
    run_id   INTEGER,
    started  REAL NOT NULL,
    ended    REAL,
    coverage REAL NOT NULL DEFAULT 0.0,
    frames   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS object_instances (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id      INTEGER NOT NULL REFERENCES places(place_id) ON DELETE CASCADE,
    cls           TEXT NOT NULL,
    semantic_kind TEXT NOT NULL DEFAULT 'unknown',
    state         TEXT NOT NULL DEFAULT 'tentative',
    anchor_x      REAL NOT NULL,
    anchor_y      REAL NOT NULL,
    width         REAL NOT NULL DEFAULT 0.1,
    height        REAL NOT NULL DEFAULT 0.1,
    tier          TEXT NOT NULL DEFAULT 'unknown',
    persistence   REAL NOT NULL DEFAULT 1.0,
    lifetime_s    REAL NOT NULL,
    ever_moved    INTEGER NOT NULL DEFAULT 0,
    n_seen        INTEGER NOT NULL DEFAULT 0,
    n_opportunities INTEGER NOT NULL DEFAULT 0,
    frames_seen   INTEGER NOT NULL DEFAULT 0,
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    observations  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_inst_place ON object_instances(place_id, cls);

-- Per-visit record. `seen: bool` alone throws away what makes a comparison
-- explainable later, so the full status and position are kept.
CREATE TABLE IF NOT EXISTS visit_observations (
    visit_id    INTEGER NOT NULL REFERENCES place_visits(id) ON DELETE CASCADE,
    instance_id INTEGER NOT NULL REFERENCES object_instances(id) ON DELETE CASCADE,
    status      TEXT NOT NULL,
    frames_seen INTEGER NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 0.0,
    anchor_x    REAL,
    anchor_y    REAL,
    PRIMARY KEY (visit_id, instance_id)
);
CREATE INDEX IF NOT EXISTS idx_visobs_inst ON visit_observations(instance_id);
"""


@dataclass
class Instance:
    id: int
    place_id: int
    cls: str
    kind: str
    state: str
    anchor: tuple[float, float]
    size: tuple[float, float]
    tier: str
    persistence: float
    lifetime_s: float
    ever_moved: bool
    n_seen: int
    n_opportunities: int
    frames_seen: int
    first_seen: float
    last_seen: float
    observations: list[tuple[float, bool]]

    def filter(self) -> PersistenceFilter:
        f = PersistenceFilter(lifetime_s=self.lifetime_s)
        for t, seen in self.observations:
            f.observe(t - self.first_seen, seen)
        return f

    def probability(self, now: float) -> float:
        return self.filter().probability(now - self.first_seen)


def _row_to_instance(r: sqlite3.Row) -> Instance:
    return Instance(
        id=r["id"], place_id=r["place_id"], cls=r["cls"], kind=r["semantic_kind"],
        state=r["state"], anchor=(r["anchor_x"], r["anchor_y"]),
        size=(r["width"], r["height"]), tier=r["tier"],
        persistence=r["persistence"], lifetime_s=r["lifetime_s"],
        ever_moved=bool(r["ever_moved"]), n_seen=r["n_seen"],
        n_opportunities=r["n_opportunities"], frames_seen=r["frames_seen"],
        first_seen=r["first_seen"], last_seen=r["last_seen"],
        observations=[tuple(o) for o in json.loads(r["observations"])],
    )


class WorldMemory:
    def __init__(
        self,
        db_path: str | Path,
        anchor_radius: float = 0.18,
        index: PlaceIndex | None = None,
        # A visit that covered almost none of the place cannot establish that
        # anything is missing. Five seconds of accidental recording must not
        # mark half the room removed.
        min_visit_coverage: float = 0.15,
        # Frames an instance must appear in before it may become permanent
        # memory. One false detection must never become "there used to be a
        # backpack here".
        min_frames_to_confirm: int = 3,
    ) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(WORLD_SCHEMA)
        self.anchor_radius = anchor_radius
        self.min_visit_coverage = min_visit_coverage
        self.min_frames_to_confirm = min_frames_to_confirm
        self.index = index or PlaceIndex()
        self._load_places()

        self.current_place: Place | None = None
        self.current_match: PlaceMatch | None = None
        self._visit_id: int | None = None
        self._seen_this_visit: set[int] = set()
        self._moved_this_visit: set[int] = set()
        self._frames_seen: Counter = Counter()
        self._confidence: dict[int, float] = {}
        self._last_anchor: dict[int, tuple[float, float]] = {}
        self._coverage: list[np.ndarray] = []
        self._occluders: list[tuple[tuple[float, float, float, float], int]] = []
        self._frames = 0

    # -- place index persistence -----------------------------------------
    def _load_places(self) -> None:
        for row in self.conn.execute("SELECT * FROM places"):
            vec = np.frombuffer(row["descriptor"], dtype=np.float32).copy()
            geo = (
                GeoFix(row["lat"], row["lon"], row["accuracy_m"] or 100.0)
                if row["lat"] is not None
                else None
            )
            place = Place(
                place_id=row["place_id"], descriptor=vec, geo=geo,
                label=row["label"], n_visits=row["n_visits"],
                first_seen_ms=int(row["first_seen"] * 1000),
                last_seen_ms=int(row["last_seen"] * 1000),
            )
            self.index.places[place.place_id] = place
            self.index._next_id = max(self.index._next_id, place.place_id + 1)

    def _save_place(self, place: Place) -> None:
        self.conn.execute(
            """INSERT INTO places
               (place_id, descriptor, dim, label, lat, lon, accuracy_m,
                n_visits, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(place_id) DO UPDATE SET
                 descriptor = excluded.descriptor,
                 label      = COALESCE(excluded.label, places.label),
                 lat        = excluded.lat, lon = excluded.lon,
                 accuracy_m = excluded.accuracy_m,
                 n_visits   = excluded.n_visits,
                 last_seen  = excluded.last_seen""",
            (
                place.place_id, place.descriptor.astype(np.float32).tobytes(),
                int(place.descriptor.size), place.label,
                place.geo.lat if place.geo else None,
                place.geo.lon if place.geo else None,
                place.geo.accuracy_m if place.geo else None,
                place.n_visits,
                place.first_seen_ms / 1000.0, place.last_seen_ms / 1000.0,
            ),
        )

    # -- visits ------------------------------------------------------------
    def begin_visit(
        self,
        image: np.ndarray,
        geo: GeoFix | None = None,
        run_id: int | None = None,
        now: float | None = None,
    ) -> PlaceMatch:
        now = now if now is not None else time.time()
        match = self.index.observe(image, int(now * 1000), geo)
        place = match.place
        self.index.begin_visit(place)
        self._save_place(place)
        cur = self.conn.execute(
            "INSERT INTO place_visits (place_id, run_id, started) VALUES (?,?,?)",
            (place.place_id, run_id, now),
        )
        self._visit_id = int(cur.lastrowid)
        self.current_place = place
        self.current_match = match
        self._seen_this_visit = set()
        self._moved_this_visit = set()
        self._frames_seen = Counter()
        self._confidence = {}
        self._last_anchor = {}
        self._coverage = []
        self._occluders = []
        self._frames = 0
        self.conn.commit()
        return match

    def observe_tracks(
        self,
        tracks: list[Track],
        now: float | None = None,
        moving_ids: set[int] | None = None,
        match: PlaceMatch | None = None,
    ) -> list[Instance]:
        """Associate visible tracks with persistent instances at this place.

        `match` may be supplied per frame -- the live path re-checks place
        identity as the camera moves; otherwise the visit's opening match is
        used.
        """
        if self.current_place is None:
            raise RuntimeError("begin_visit() must be called first")
        now = now if now is not None else time.time()
        moving_ids = moving_ids or set()
        match = match or self.current_match
        self._frames += 1

        # Coverage: the region of the canonical frame this view actually saw.
        if match is not None:
            corners = match.frame_corners_canonical()
            if corners is not None:
                self._coverage.append(corners.astype(np.float32))

        can_warp = match is not None and match.homography is not None
        existing = self._instances(self.current_place.place_id)
        out: list[Instance] = []

        for track in tracks:
            raw = track.centroid()
            warped = match.to_canonical(*raw) if can_warp else None
            anchor = warped if warped is not None else raw
            trusted = warped is not None

            inst = self._associate(
                existing, track, anchor,
                radius=self.anchor_radius if trusted else self.anchor_radius * 0.5,
            )
            if inst is None:
                # Without a trustworthy transform we cannot say *where* in the
                # place this is, so it must not become permanent memory -- that
                # is precisely how duplicate instances accumulate.
                if not trusted:
                    continue
                inst = self._create(track, anchor, now)
                existing.append(inst)

            if track.track_id in moving_ids:
                # Recorded per visit rather than on `inst`, because end_visit()
                # reloads instances from the database and would discard a
                # mutation made here.
                self._moved_this_visit.add(inst.id)
            self._seen_this_visit.add(inst.id)
            self._frames_seen[inst.id] += 1
            self._confidence[inst.id] = max(
                self._confidence.get(inst.id, 0.0), track.last.score
            )
            self._last_anchor[inst.id] = anchor

            # Anything actually detected can occlude something behind it.
            box = self._canonical_box(track, match if trusted else None)
            if box is not None:
                self._occluders.append((box, inst.id))
            out.append(inst)
        return out

    def end_visit(self, now: float | None = None) -> dict:
        """Close the visit, recording what was established and what was not.

        Idempotent: explicit stop and socket disconnect can both fire, and a
        second call must not append a second round of evidence.
        """
        if self.current_place is None or self._visit_id is None:
            return {}
        now = now if now is not None else time.time()
        place_id = self.current_place.place_id
        visit_id = self._visit_id
        coverage = self._coverage_fraction()
        established = coverage >= self.min_visit_coverage

        counts: Counter = Counter()
        discarded = 0
        for inst in self._instances(place_id):
            status = self._status(inst, established)
            counts[status] += 1

            if inst.id in self._moved_this_visit:
                inst.ever_moved = True

            # Only established evidence reaches the filter. `not_observed` and
            # `occluded` append nothing, so belief decays under the survival
            # prior alone -- exactly what "I haven't checked" should mean.
            if is_evidence(status):
                seen = status == ObservationStatus.SEEN
                inst.observations.append((now, seen))
                inst.n_opportunities += 1
                if seen:
                    inst.n_seen += 1
                    inst.last_seen = now

            inst.frames_seen += self._frames_seen.get(inst.id, 0)

            # Lifecycle: a tentative instance that never accumulated enough
            # frames is deleted, not remembered as having been removed.
            #
            # The discard test is "did it have a fair chance?". If the visit
            # ran for at least as many frames as confirmation requires and the
            # object still registered in fewer, that is evidence of a spurious
            # detection rather than of a real object. Being seen once in six
            # frames is exactly what a detector hallucination looks like.
            if inst.state == InstanceState.TENTATIVE:
                if inst.frames_seen >= self.min_frames_to_confirm:
                    inst.state = InstanceState.CONFIRMED
                elif (
                    status != ObservationStatus.SEEN
                    or self._frames >= self.min_frames_to_confirm
                ):
                    self.conn.execute(
                        "DELETE FROM object_instances WHERE id = ?", (inst.id,)
                    )
                    self.conn.execute(
                        "DELETE FROM visit_observations WHERE instance_id = ?",
                        (inst.id,),
                    )
                    discarded += 1
                    continue

            inst.persistence = inst.probability(now)
            inst.tier = classify(
                n_visits=inst.n_opportunities,
                hit_rate=inst.n_seen / max(1, inst.n_opportunities),
                kind=inst.kind,
                observed_moving=inst.ever_moved,
            )
            if inst.id in self._last_anchor:
                inst.anchor = self._last_anchor[inst.id]
            self._save_instance(inst)
            self._record_visit_observation(visit_id, inst, status)

        self.conn.execute(
            "UPDATE place_visits SET ended = ?, coverage = ?, frames = ? WHERE id = ?",
            (now, coverage, self._frames, visit_id),
        )
        self.conn.commit()

        summary = {
            "place_id": place_id,
            "visit_id": visit_id,
            "n_visits": self.current_place.n_visits,
            "coverage": round(coverage, 4),
            "coverage_established": established,
            "frames": self._frames,
            "seen": counts[ObservationStatus.SEEN],
            "checked_missing": counts[ObservationStatus.CHECKED_MISSING],
            "occluded": counts[ObservationStatus.OCCLUDED_UNCERTAIN],
            "not_observed": counts[ObservationStatus.NOT_OBSERVED],
            "discarded_tentative": discarded,
        }
        self.current_place = None
        self.current_match = None
        self._visit_id = None
        return summary

    # -- observation status -------------------------------------------------
    def _status(self, inst: Instance, established: bool) -> str:
        if inst.id in self._seen_this_visit:
            return ObservationStatus.SEEN
        if not established or not self._covered(*inst.anchor):
            # Either too little of the place was covered to conclude anything,
            # or this particular spot was never in view.
            return ObservationStatus.NOT_OBSERVED
        if self._occluded(inst):
            return ObservationStatus.OCCLUDED_UNCERTAIN
        return ObservationStatus.CHECKED_MISSING

    def _covered(self, x: float, y: float) -> bool:
        pt = (float(x), float(y))
        return any(
            cv2.pointPolygonTest(quad, pt, False) >= 0 for quad in self._coverage
        )

    def _occluded(self, inst: Instance) -> bool:
        """Was something in the way?

        Deliberately generous: any detected object covering the expected
        position counts. Being wrong here costs an unresolved "not confirmed";
        being wrong the other way reports a removal that never happened.
        """
        x, y = inst.anchor
        for (x1, y1, x2, y2), owner in self._occluders:
            if owner == inst.id:
                continue
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _coverage_fraction(self) -> float:
        """Fraction of the canonical unit square observed during this visit."""
        if not self._coverage:
            return 0.0
        grid = 32
        mask = np.zeros((grid, grid), dtype=np.uint8)
        for quad in self._coverage:
            pts = np.clip(quad, 0.0, 1.0) * (grid - 1)
            cv2.fillConvexPoly(mask, pts.astype(np.int32), 1)
        return float(mask.mean())

    def _canonical_box(
        self, track: Track, match: PlaceMatch | None
    ) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = track.last.box
        if match is None:
            return (x1, y1, x2, y2)
        a = match.to_canonical(x1, y1)
        b = match.to_canonical(x2, y2)
        if a is None or b is None:
            return None
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    # -- association --------------------------------------------------------
    def _associate(
        self,
        instances: list[Instance],
        track: Track,
        anchor: tuple[float, float],
        radius: float,
    ) -> Instance | None:
        """Match a track to an existing instance.

        Position alone is too weak: two chairs side by side swap identities on
        any small anchor error. Geometry is folded in so a large object cannot
        absorb a small one sitting at the same spot. Class is a hard gate
        rather than a soft cost -- a `chair` is never a `tv`, however close.
        """
        x1, y1, x2, y2 = track.last.box
        tw, th = max(1e-6, abs(x2 - x1)), max(1e-6, abs(y2 - y1))
        best, best_score = None, 0.0
        for inst in instances:
            if inst.cls != track.cls:
                continue
            d = float(np.hypot(inst.anchor[0] - anchor[0], inst.anchor[1] - anchor[1]))
            if d > radius:
                continue
            spatial = 1.0 - (d / radius)
            iw, ih = max(1e-6, inst.size[0]), max(1e-6, inst.size[1])
            area_ratio = (tw * th) / (iw * ih)
            aspect_ratio = (tw / th) / (iw / ih)
            # Both ratios peak at 1.0 and are penalised symmetrically.
            geometry = (min(area_ratio, 1 / area_ratio) ** 0.5) * min(
                aspect_ratio, 1 / aspect_ratio
            )
            score = 0.7 * spatial + 0.3 * geometry
            if score > best_score:
                best, best_score = inst, score
        return best

    def _create(
        self, track: Track, anchor: tuple[float, float], now: float
    ) -> Instance:
        cls = track.cls
        lifetime = lifetime_prior(cls)
        kind = semantic_kind(cls)
        x1, y1, x2, y2 = track.last.box
        w, h = abs(x2 - x1), abs(y2 - y1)
        cur = self.conn.execute(
            """INSERT INTO object_instances
               (place_id, cls, semantic_kind, state, anchor_x, anchor_y,
                width, height, tier, persistence, lifetime_s,
                first_seen, last_seen, observations)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'[]')""",
            (
                self.current_place.place_id, cls, kind, InstanceState.TENTATIVE,
                anchor[0], anchor[1], w, h, Tier.UNKNOWN, 1.0, lifetime, now, now,
            ),
        )
        return Instance(
            id=int(cur.lastrowid), place_id=self.current_place.place_id,
            cls=cls, kind=kind, state=InstanceState.TENTATIVE, anchor=anchor,
            size=(w, h), tier=Tier.UNKNOWN, persistence=1.0,
            lifetime_s=lifetime, ever_moved=False, n_seen=0, n_opportunities=0,
            frames_seen=0, first_seen=now, last_seen=now, observations=[],
        )

    def _save_instance(self, inst: Instance) -> None:
        self.conn.execute(
            """UPDATE object_instances SET
                 state = ?, tier = ?, persistence = ?, ever_moved = ?,
                 n_seen = ?, n_opportunities = ?, frames_seen = ?,
                 anchor_x = ?, anchor_y = ?, last_seen = ?, observations = ?
               WHERE id = ?""",
            (
                inst.state, inst.tier, inst.persistence, int(inst.ever_moved),
                inst.n_seen, inst.n_opportunities, inst.frames_seen,
                inst.anchor[0], inst.anchor[1], inst.last_seen,
                json.dumps(inst.observations), inst.id,
            ),
        )

    def _record_visit_observation(
        self, visit_id: int, inst: Instance, status: str
    ) -> None:
        anchor = self._last_anchor.get(inst.id)
        self.conn.execute(
            """INSERT INTO visit_observations
               (visit_id, instance_id, status, frames_seen, confidence,
                anchor_x, anchor_y)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(visit_id, instance_id) DO UPDATE SET
                 status = excluded.status, frames_seen = excluded.frames_seen,
                 confidence = excluded.confidence,
                 anchor_x = excluded.anchor_x, anchor_y = excluded.anchor_y""",
            (
                visit_id, inst.id, status, self._frames_seen.get(inst.id, 0),
                self._confidence.get(inst.id, 0.0),
                anchor[0] if anchor else None, anchor[1] if anchor else None,
            ),
        )

    def _instances(
        self, place_id: int, confirmed_only: bool = False
    ) -> list[Instance]:
        sql = "SELECT * FROM object_instances WHERE place_id = ?"
        args: list = [place_id]
        if confirmed_only:
            sql += " AND state != ?"
            args.append(InstanceState.TENTATIVE)
        return [_row_to_instance(r) for r in self.conn.execute(sql, args)]

    # -- queries -----------------------------------------------------------
    def snapshot(self, place_id: int, now: float | None = None) -> list[dict]:
        """What the system believes is at a place right now."""
        now = now if now is not None else time.time()
        out = [
            {
                "instance_id": i.id,
                "class": i.cls,
                "semantic_kind": i.kind,
                "state": i.state,
                "tier": i.tier,
                "persistence": round(i.probability(now), 4),
                "anchor": [round(i.anchor[0], 3), round(i.anchor[1], 3)],
                "times_seen": i.n_seen,
                "opportunities": i.n_opportunities,
                "last_seen": i.last_seen,
            }
            for i in self._instances(place_id, confirmed_only=True)
        ]
        out.sort(key=lambda i: i["persistence"], reverse=True)
        return out

    def changes(
        self, place_id: int, threshold: float = 0.5, now: float | None = None
    ) -> dict:
        """What changed at a place, against accumulated belief.

        People and animals are excluded: someone walking through is not a
        change to the world, and reporting them would bury the real changes.
        """
        now = now if now is not None else time.time()
        gone, arrived, stable = [], [], []
        for inst in self._instances(place_id, confirmed_only=True):
            if inst.tier == Tier.DYNAMIC or inst.kind in (
                SemanticKind.PERSON, SemanticKind.ANIMAL
            ):
                continue
            p = inst.probability(now)
            entry = {
                "instance_id": inst.id, "class": inst.cls,
                "semantic_kind": inst.kind, "tier": inst.tier,
                "persistence": round(p, 4), "last_seen": inst.last_seen,
                "anchor": [round(inst.anchor[0], 3), round(inst.anchor[1], 3)],
            }
            # "Was established, then absent" -- something seen once and never
            # again was never established enough to count as removed.
            if p < threshold and inst.n_seen >= 2:
                gone.append(entry)
            elif p >= threshold and inst.n_opportunities <= 2:
                arrived.append(entry)
            elif p >= threshold:
                stable.append(entry)
        return {"removed": gone, "added": arrived, "unchanged": stable}

    def find(self, cls: str, now: float | None = None) -> list[dict]:
        """Where was this last seen? Ranked by belief it is still there."""
        now = now if now is not None else time.time()
        rows = self.conn.execute(
            """SELECT i.*, p.label FROM object_instances i
               JOIN places p ON p.place_id = i.place_id
               WHERE i.cls = ? AND i.state != ?
               ORDER BY i.last_seen DESC""",
            (cls, InstanceState.TENTATIVE),
        ).fetchall()
        out = [
            {
                "instance_id": r["id"], "place_id": r["place_id"],
                "place_label": r["label"], "class": r["cls"],
                "semantic_kind": r["semantic_kind"], "tier": r["tier"],
                "still_there": round(_row_to_instance(r).probability(now), 4),
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]
        out.sort(key=lambda i: i["still_there"], reverse=True)
        return out

    def visit_observations(self, visit_id: int) -> dict[int, dict]:
        """Per-instance record of one visit, keyed by instance id.

        This is what makes a later comparison auditable: the difference
        between "checked and gone" and "never looked" is preserved here rather
        than collapsed into a boolean.
        """
        rows = self.conn.execute(
            """SELECT vo.*, i.cls, i.semantic_kind FROM visit_observations vo
               JOIN object_instances i ON i.id = vo.instance_id
               WHERE vo.visit_id = ?""",
            (visit_id,),
        ).fetchall()
        return {
            r["instance_id"]: {
                "instance_id": r["instance_id"],
                "class": r["cls"],
                "semantic_kind": r["semantic_kind"],
                "status": r["status"],
                "frames_seen": r["frames_seen"],
                "confidence": r["confidence"],
                "anchor": (
                    [r["anchor_x"], r["anchor_y"]]
                    if r["anchor_x"] is not None
                    else None
                ),
            }
            for r in rows
        }

    def visits(self, place_id: int, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM place_visits WHERE place_id = ? AND ended IS NOT NULL
               ORDER BY started DESC LIMIT ?""",
            (place_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def label_place(self, place_id: int, label: str) -> None:
        self.conn.execute(
            "UPDATE places SET label = ? WHERE place_id = ?", (label, place_id)
        )
        if place_id in self.index.places:
            self.index.places[place_id].label = label
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "WorldMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
