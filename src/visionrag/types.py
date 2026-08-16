"""Core data contracts.

These types are the paper's contribution: a typed, time-indexed memory
substrate built without a VLM in the ingest loop. Treat this module as a
schema document -- changes here invalidate stored runs, so bump
``SCHEMA_VERSION`` whenever a persisted field changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1

# Boxes are normalised xyxy in [0, 1] throughout the pipeline. They are
# converted to xywh only at the API boundary (see PRD 7.1). Normalised
# coordinates keep events resolution-independent, which matters because the
# scheduler changes analysis resolution under load.
Box = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    """One model observation in one frame. No temporal identity yet."""

    cls: str
    score: float
    box: Box
    feature: list[float] | None = None  # appearance embedding, only if computed


@dataclass(slots=True)
class TrackState:
    """A track's observed state at one analysed frame."""

    frame_id: int
    ts_ms: int
    box: Box
    score: float
    # Velocity in normalised units per second. `raw` is image-space motion;
    # `compensated` has estimated camera motion removed. The gap between them
    # is what the ego-motion counterfactual gate tests (claim S1).
    v_raw: tuple[float, float] = (0.0, 0.0)
    v_compensated: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class Track:
    """Temporal identity of one object across frames."""

    track_id: int
    cls: str
    first_seen_ms: int
    last_seen_ms: int
    states: list[TrackState] = field(default_factory=list)
    # Frames since last successful match. The tracker keeps a track alive
    # through short occlusions rather than fragmenting its identity.
    misses: int = 0
    confirmed: bool = False

    @property
    def last(self) -> TrackState:
        return self.states[-1]

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.last.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class EventType(str, Enum):
    """Event taxonomy (PRD Appendix A).

    Risk-tier events are deliberately absent: they belong to a later phase and
    are out of scope for the first paper.
    """

    # Visibility
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    REAPPEARED = "reappeared"
    # Motion
    STARTED_MOVING = "started_moving"
    STOPPED = "stopped"
    DIRECTION_CHANGED = "direction_changed"
    # Spatial
    ENTERED_REGION = "entered_region"
    LEFT_REGION = "left_region"
    NEAR = "near"
    # Interaction
    APPROACHED = "approached"
    MOVED_AWAY = "moved_away"
    REMAINED_NEAR = "remained_near"
    # Scene
    CAMERA_MOVED = "camera_moved"
    CAMERA_STABILISED = "camera_stabilised"


@dataclass(slots=True)
class Event:
    """An interpreted temporal relation over one or more tracks.

    An Event is a *hypothesis*, never a claim about intent or causation
    (PRD 6.2). `confidence` must survive into the answer layer so that
    abstention is possible.
    """

    type: EventType
    t_start_ms: int
    t_end_ms: int
    participants: list[int]  # track_ids
    confidence: float
    evidence_frames: list[int] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    # Set when the ego-motion counterfactual could not rule out camera motion
    # as an explanation. Such events are retained but down-weighted, never
    # silently dropped -- suppression rates are themselves a reported metric.
    ego_suspect: bool = False

    def duration_ms(self) -> int:
        return self.t_end_ms - self.t_start_ms


@dataclass(slots=True)
class CameraMotion:
    """Estimated global frame-to-frame camera motion."""

    dx: float  # normalised translation
    dy: float
    scale: float
    rotation: float  # radians
    inliers: int
    ok: bool  # False when estimation failed (too few features)

    @property
    def magnitude(self) -> float:
        return (self.dx**2 + self.dy**2) ** 0.5


@dataclass(slots=True)
class FrameResult:
    """What the pipeline emits per analysed frame."""

    frame_id: int
    ts_ms: int
    detections: list[Detection]
    tracks: list[Track]
    events: list[Event]
    camera: CameraMotion | None
    detector_ran: bool  # False when the scheduler skipped detection
