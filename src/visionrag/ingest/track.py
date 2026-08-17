"""Multi-object tracking.

A SORT-style tracker: greedy IoU association with a constant-velocity motion
prior. Appearance features are deliberately not used by default -- they are
the single most expensive optional component, and the paper's claim is about
what can be achieved *without* per-frame neural work beyond detection.

Velocities are recorded twice, raw and ego-compensated, because the difference
between them is the evidence for claim S1.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerConfig
from ..types import Box, CameraMotion, Detection, Track, TrackState


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def _centroid(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def active(self) -> list[Track]:
        """Confirmed tracks matched in the most recent update."""
        return [t for t in self._tracks.values() if t.confirmed and t.misses == 0]

    def reset(self) -> None:
        """Drop all tracks, keeping the id counter monotonic.

        Used when the camera moves to a different place: tracks describe
        objects in the room they were seen in, and carrying them across a
        transition would attach one room's objects to another's. Ids are not
        reused, so stored observations referring to old track ids stay
        unambiguous.
        """
        self._tracks.clear()

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        ts_ms: int,
        camera: CameraMotion | None = None,
    ) -> tuple[list[Track], list[Track]]:
        """Associate detections to tracks.

        Returns ``(newly_confirmed, lost)`` so the event layer can emit
        appeared/disappeared without re-deriving track lifecycle itself.
        """
        # Association is class-aware: two objects of different classes are
        # never the same object, and letting them compete produces ID switches
        # that inflate the fragmentation metric for no reason.
        unmatched_dets = set(range(len(detections)))
        matched: dict[int, int] = {}  # track_id -> detection index

        candidates = [
            (tid, t) for tid, t in self._tracks.items() if t.misses <= self.cfg.max_misses
        ]
        # Match longer-lived tracks first; they carry more evidence, so on a
        # contested detection they should win.
        candidates.sort(key=lambda kv: len(kv[1].states), reverse=True)

        for tid, track in candidates:
            best_i, best_iou = -1, self.cfg.iou_thresh
            predicted = self._predict(track, ts_ms)
            for i in unmatched_dets:
                det = detections[i]
                if det.cls != track.cls:
                    continue
                score = iou(predicted, det.box)
                if score > best_iou:
                    best_i, best_iou = i, score
            if best_i >= 0:
                matched[tid] = best_i
                unmatched_dets.discard(best_i)

        newly_confirmed: list[Track] = []

        for tid, det_i in matched.items():
            track = self._tracks[tid]
            det = detections[det_i]
            state = self._make_state(track, det, frame_id, ts_ms, camera)
            track.states.append(state)
            track.last_seen_ms = ts_ms
            track.misses = 0
            if not track.confirmed and len(track.states) >= self.cfg.min_hits:
                track.confirmed = True
                newly_confirmed.append(track)

        for tid, track in self._tracks.items():
            if tid not in matched:
                track.misses += 1

        for i in unmatched_dets:
            if len(self._tracks) >= self.cfg.max_tracks:
                break
            det = detections[i]
            track = Track(
                track_id=self._next_id,
                cls=det.cls,
                first_seen_ms=ts_ms,
                last_seen_ms=ts_ms,
                states=[
                    TrackState(
                        frame_id=frame_id, ts_ms=ts_ms, box=det.box, score=det.score
                    )
                ],
            )
            self._tracks[self._next_id] = track
            self._next_id += 1
            if self.cfg.min_hits <= 1:
                track.confirmed = True
                newly_confirmed.append(track)

        lost = [
            t
            for t in self._tracks.values()
            if t.misses > self.cfg.max_misses and t.confirmed
        ]
        for t in lost:
            self._tracks.pop(t.track_id, None)

        return newly_confirmed, lost

    def _predict(self, track: Track, ts_ms: int) -> Box:
        """Constant-velocity box prediction. Keeps association stable when a
        fast object moves far enough between analysed frames that its raw IoU
        with the previous box would be zero -- the usual cause of ID switches
        at low analysis frame rates, which is exactly the regime we run in."""
        if len(track.states) < 2:
            return track.last.box
        dt = (ts_ms - track.last.ts_ms) / 1000.0
        if dt <= 0:
            return track.last.box
        vx, vy = track.last.v_raw
        x1, y1, x2, y2 = track.last.box
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    def _make_state(
        self,
        track: Track,
        det: Detection,
        frame_id: int,
        ts_ms: int,
        camera: CameraMotion | None,
    ) -> TrackState:
        v_raw = (0.0, 0.0)
        v_comp = (0.0, 0.0)
        if track.states:
            prev = track.last
            dt = (ts_ms - prev.ts_ms) / 1000.0
            if dt > 1e-6:
                cx, cy = _centroid(det.box)
                px, py = _centroid(prev.box)
                v_raw = ((cx - px) / dt, (cy - py) / dt)
                if camera is not None and camera.ok:
                    # Subtract the image-space displacement the camera alone
                    # would have produced. What remains is motion attributable
                    # to the object.
                    v_comp = (
                        v_raw[0] - camera.dx / dt,
                        v_raw[1] - camera.dy / dt,
                    )
                else:
                    v_comp = v_raw
        return TrackState(
            frame_id=frame_id,
            ts_ms=ts_ms,
            box=det.box,
            score=det.score,
            v_raw=v_raw,
            v_compensated=v_comp,
        )
