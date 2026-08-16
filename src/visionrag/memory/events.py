"""Event induction.

This is the substrate the paper argues for: instead of captioning frames with
a VLM and embedding the prose, derive *typed* events from geometry and
temporal state. Events carry participants, time bounds, evidence frames and
calibrated confidence, which makes them retrievable by structure rather than
by vibe -- and cheap enough to run on a CPU.

Rules are deliberately deterministic (PRD 6.2). Every rule emits a
*hypothesis*, never a claim about intent or causation: "approached" is a
geometric fact about distance, and nothing here should be read as "wanted to
reach".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import EgoMotionConfig, EventConfig
from ..types import CameraMotion, Event, EventType, Track


def _centroid(track: Track) -> tuple[float, float]:
    return track.centroid()


def _distance(a: Track, b: Track) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return float(np.hypot(ax - bx, ay - by))


@dataclass
class _PairState:
    near: bool = False
    near_since_ms: int | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=64))  # (ts_ms, dist)
    # Direction of the trend episode currently in progress, if any. One event
    # is emitted per episode rather than once per window: a four-second
    # approach is one approach, and re-emitting it every window would bloat
    # the store and skew retrieval toward whichever pair moved longest.
    trend_kind: EventType | None = None


@dataclass
class _MotionState:
    moving: bool = False
    since_ms: int | None = None


class EventInducer:
    """Stateful rule engine. One instance per session."""

    def __init__(self, cfg: EventConfig, ego_cfg: EgoMotionConfig) -> None:
        self.cfg = cfg
        self.ego_cfg = ego_cfg
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self._motion: dict[int, _MotionState] = {}
        self._camera_moving = False
        self._seen_before: set[int] = set()

    # -- confidence -------------------------------------------------------
    def _confidence(
        self,
        base: float,
        tracks: list[Track],
        persistence: float = 1.0,
        ego_suspect: bool = False,
    ) -> float:
        """Carry detector uncertainty forward and reward temporal consistency
        (PRD 6.3). Deliberately conservative: an over-confident event layer
        makes the calibration results in E4 look worse, not better."""
        det = float(np.mean([t.last.score for t in tracks])) if tracks else 0.5
        # Short-lived tracks are less trustworthy; saturates around 10 states.
        evidence = min(1.0, np.mean([len(t.states) for t in tracks]) / 10.0) if tracks else 0.5
        conf = base * (0.5 + 0.5 * det) * (0.6 + 0.4 * evidence) * persistence
        if ego_suspect:
            # Not zeroed: the event is still reported, flagged, and counted.
            conf *= 0.4
        return float(np.clip(conf, 0.0, 1.0))

    # -- main entry -------------------------------------------------------
    def process(
        self,
        active: list[Track],
        newly_confirmed: list[Track],
        lost: list[Track],
        frame_id: int,
        ts_ms: int,
        camera: CameraMotion | None,
    ) -> list[Event]:
        events: list[Event] = []
        events += self._visibility(newly_confirmed, lost, frame_id, ts_ms)
        events += self._camera(camera, frame_id, ts_ms)
        events += self._motion_events(active, frame_id, ts_ms, camera)
        events += self._spatial(active, frame_id, ts_ms)
        return [e for e in events if e.confidence >= self.cfg.min_confidence]

    # -- rules ------------------------------------------------------------
    def _visibility(
        self, newly_confirmed: list[Track], lost: list[Track], frame_id: int, ts_ms: int
    ) -> list[Event]:
        out: list[Event] = []
        for t in newly_confirmed:
            # A track id is never reused, so "reappeared" here means a track
            # this inducer previously saw lost -- which the tracker cannot
            # express on its own. Kept distinct because re-identification
            # failures are a known limitation worth measuring, not hiding.
            kind = (
                EventType.REAPPEARED
                if t.track_id in self._seen_before
                else EventType.APPEARED
            )
            self._seen_before.add(t.track_id)
            out.append(
                Event(
                    type=kind,
                    t_start_ms=t.first_seen_ms,
                    t_end_ms=ts_ms,
                    participants=[t.track_id],
                    confidence=self._confidence(0.9, [t]),
                    evidence_frames=[frame_id],
                    attrs={"class": t.cls},
                )
            )
        for t in lost:
            out.append(
                Event(
                    type=EventType.DISAPPEARED,
                    t_start_ms=t.last_seen_ms,
                    t_end_ms=ts_ms,
                    participants=[t.track_id],
                    confidence=self._confidence(0.8, [t]),
                    evidence_frames=[t.last.frame_id],
                    attrs={"class": t.cls},
                )
            )
            self._motion.pop(t.track_id, None)
            for key in [k for k in self._pairs if t.track_id in k]:
                self._pairs.pop(key, None)
        return out

    def _camera(
        self, camera: CameraMotion | None, frame_id: int, ts_ms: int
    ) -> list[Event]:
        if camera is None or not camera.ok:
            return []
        moving = camera.magnitude >= self.ego_cfg.static_thresh
        if moving == self._camera_moving:
            return []
        self._camera_moving = moving
        kind = EventType.CAMERA_MOVED if moving else EventType.CAMERA_STABILISED
        return [
            Event(
                type=kind,
                t_start_ms=ts_ms,
                t_end_ms=ts_ms,
                participants=[],
                confidence=min(1.0, 0.5 + camera.inliers / 200.0),
                evidence_frames=[frame_id],
                attrs={"magnitude": round(camera.magnitude, 5)},
            )
        ]

    def _motion_events(
        self,
        active: list[Track],
        frame_id: int,
        ts_ms: int,
        camera: CameraMotion | None,
    ) -> list[Event]:
        out: list[Event] = []
        for t in active:
            if len(t.states) < 2:
                continue
            state = self._motion.setdefault(t.track_id, _MotionState())
            v_raw = t.last.v_raw
            v_comp = t.last.v_compensated
            # Speed is judged on compensated velocity: the question is whether
            # the object moved in the world, not whether its pixels moved.
            speed = float(np.hypot(*v_comp))
            moving = speed >= self.cfg.stop_speed_thresh

            ego_suspect = False
            if camera is not None and camera.ok and moving:
                raw_speed = float(np.hypot(*v_raw))
                if raw_speed > 1e-6:
                    ego_suspect = (
                        speed / raw_speed
                    ) < self.ego_cfg.counterfactual_ratio

            if state.since_ms is None:
                state.moving, state.since_ms = moving, ts_ms
                continue
            if moving == state.moving:
                continue

            duration_s = (ts_ms - state.since_ms) / 1000.0
            # Require the *previous* state to have held long enough, so
            # detector jitter does not produce alternating start/stop pairs.
            if duration_s < self.cfg.stop_min_duration_s:
                continue

            kind = EventType.STARTED_MOVING if moving else EventType.STOPPED
            # Transition events are stamped at the instant of transition, not
            # over the span of the state that preceded them. "When did it
            # stop?" must answer with the moment it stopped; the preceding
            # duration is evidence for the event, not its extent.
            out.append(
                Event(
                    type=kind,
                    t_start_ms=ts_ms,
                    t_end_ms=ts_ms,
                    participants=[t.track_id],
                    confidence=self._confidence(
                        0.75, [t], min(1.0, duration_s), ego_suspect
                    ),
                    evidence_frames=[frame_id],
                    attrs={
                        "class": t.cls,
                        "speed": round(speed, 5),
                        "prior_state_s": round(duration_s, 3),
                    },
                    ego_suspect=ego_suspect,
                )
            )
            state.moving, state.since_ms = moving, ts_ms
        return out

    def _spatial(self, active: list[Track], frame_id: int, ts_ms: int) -> list[Event]:
        out: list[Event] = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                pair = self._pairs.setdefault(key, _PairState())
                dist = _distance(a, b)
                pair.history.append((ts_ms, dist))

                # near / remained_near, with hysteresis so an object sitting
                # exactly at the threshold does not emit an event per frame.
                if not pair.near and dist <= self.cfg.near_thresh:
                    pair.near, pair.near_since_ms = True, ts_ms
                    out.append(
                        Event(
                            type=EventType.NEAR,
                            t_start_ms=ts_ms,
                            t_end_ms=ts_ms,
                            participants=list(key),
                            confidence=self._confidence(0.85, [a, b]),
                            evidence_frames=[frame_id],
                            attrs={"distance": round(dist, 4)},
                        )
                    )
                elif pair.near and dist > self.cfg.far_thresh:
                    dwell_s = (ts_ms - (pair.near_since_ms or ts_ms)) / 1000.0
                    if dwell_s >= self.cfg.remained_near_min_s:
                        out.append(
                            Event(
                                type=EventType.REMAINED_NEAR,
                                t_start_ms=pair.near_since_ms or ts_ms,
                                t_end_ms=ts_ms,
                                participants=list(key),
                                confidence=self._confidence(
                                    0.8, [a, b], min(1.0, dwell_s / 5.0)
                                ),
                                evidence_frames=[frame_id],
                                attrs={"duration_s": round(dwell_s, 2)},
                            )
                        )
                    pair.near, pair.near_since_ms = False, None

                trend = self._trend(pair, ts_ms)
                if trend is None:
                    # Episode broken -- re-arm so a later approach can fire.
                    pair.trend_kind = None
                else:
                    delta, kind, window_start_ms = trend
                    if kind != pair.trend_kind:
                        pair.trend_kind = kind
                        # Emitted at episode start rather than at its end, so
                        # live mode surfaces the event as soon as it is
                        # supportable instead of waiting for the motion to
                        # finish.
                        out.append(
                            Event(
                                type=kind,
                                t_start_ms=window_start_ms,
                                t_end_ms=ts_ms,
                                participants=list(key),
                                confidence=self._confidence(0.7, [a, b]),
                                evidence_frames=[frame_id],
                                attrs={
                                    "delta": round(delta, 4),
                                    "distance": round(dist, 4),
                                },
                            )
                        )
        return out

    def _trend(
        self, pair: _PairState, ts_ms: int
    ) -> tuple[float, EventType, int] | None:
        """Monotone distance change over the configured window.

        Requires consistency across the whole window rather than just
        comparing endpoints -- two objects that pass each other would
        otherwise register as a clean approach.
        """
        window_ms = self.cfg.approach_window_s * 1000
        pts = [(t, d) for t, d in pair.history if ts_ms - t <= window_ms]
        if len(pts) < 3:
            return None
        dists = [d for _, d in pts]
        delta = dists[-1] - dists[0]
        if abs(delta) < self.cfg.approach_min_delta:
            return None
        diffs = np.diff(dists)
        # At least 70% of steps must agree with the overall direction.
        agree = float(np.mean(diffs < 0)) if delta < 0 else float(np.mean(diffs > 0))
        if agree < 0.7:
            return None
        # Window start comes from observed history, not `ts_ms - window`,
        # which runs before the start of the session near t=0 and produces
        # negative timestamps.
        return (
            delta,
            EventType.APPROACHED if delta < 0 else EventType.MOVED_AWAY,
            pts[0][0],
        )
