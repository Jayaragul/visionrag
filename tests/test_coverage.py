"""Observation coverage and occlusion.

Gates 5 and 6: looking at half a room must not make the other half disappear,
and something standing in the way must not be reported as a removal.

These use real geometric verification (not the identity fallback), because
coverage is derived from the homography that maps the live frame into the
place's canonical frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionrag.memory.persistence import (  # noqa: E402
    SECONDS_PER_DAY,
    InstanceState,
    ObservationStatus,
)
from visionrag.memory.places import PlaceIndex  # noqa: E402
from visionrag.memory.world import WorldMemory  # noqa: E402
from visionrag.types import Track, TrackState  # noqa: E402

DAY = SECONDS_PER_DAY
T0 = 1_760_000_000.0


def track(tid: int, cls: str, cx: float, cy: float, half: float = 0.05) -> Track:
    box = (cx - half, cy - half, cx + half, cy + half)
    return Track(
        tid, cls, 0, 1000,
        states=[TrackState(frame_id=0, ts_ms=1000, box=box, score=0.9)],
        confirmed=True,
    )


def textured_scene(seed: int, size=(640, 480)) -> np.ndarray:
    """A feature-rich, *non-repetitive* scene.

    ORB needs corners, but it must not be given a regular grid: repeated
    texture makes every intersection look like every other, and RANSAC will
    happily fit a large, tight consensus to a cluster of wrong matches. That
    produced a homography collapsing the whole frame to a point while
    reporting 37 inliers at 0.005 error -- which is why `_is_plausible` exists.

    Shapes here vary in size, colour and orientation so correspondences are
    unambiguous, and survive the 2x rescale a zoomed view applies.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    for _ in range(26):
        x, y = int(rng.integers(0, w - 90)), int(rng.integers(0, h - 90))
        bw, bh = int(rng.integers(24, 88)), int(rng.integers(24, 88))
        colour = tuple(int(c) for c in rng.integers(40, 255, 3))
        if rng.random() < 0.35:
            cv2.circle(img, (x + bw // 2, y + bh // 2), bw // 2, colour, -1)
        else:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)
        # An off-axis stroke gives corners at an angle no other shape repeats.
        cv2.line(
            img, (x, y + bh), (x + bw, y),
            tuple(int(c) for c in rng.integers(0, 255, 3)), 2,
        )
    return img


def panorama(seed: int, width: int = 1280, height: int = 480) -> np.ndarray:
    """A place wider than any single camera view."""
    return textured_scene(seed, size=(width, height))


def view(pano: np.ndarray, offset_px: int, width: int = 640) -> np.ndarray:
    """A camera-sized window onto the panorama.

    Panning rather than zooming: a shifted window keeps scale constant, which
    is both what turning your head actually does and what ORB matches most
    reliably. A 2x zoom of the same scene changes the tiled histogram enough
    that the place is no longer recognised at all -- a real limitation of this
    descriptor, and one that belongs in the Phase 1 validation on real photos
    rather than being papered over here.
    """
    return pano[:, offset_px : offset_px + width].copy()


@pytest.fixture()
def world(tmp_path):
    with WorldMemory(
        tmp_path / "world.db", index=PlaceIndex(verify_geometry=True)
    ) as w:
        yield w


def establish(world, scene, objects, days=(0, 1, 2), frames=4):
    """Get instances confirmed by seeing them over several full-view visits."""
    for day in days:
        now = T0 + day * DAY
        world.begin_visit(scene, now=now)
        for _ in range(frames):
            world.observe_tracks(objects, now=now)
        world.end_visit(now=now)


def test_degenerate_homography_is_rejected():
    """A well-fitted but geometrically nonsensical transform must be refused.

    Inlier count and reprojection error cannot detect a collapsed or mirrored
    homography, so the plausibility check is what stands between the system
    and confidently placing every object at the same point.
    """
    index = PlaceIndex()
    # Scale small enough to collapse the frame, but not so small that the
    # singularity check catches it first -- this is the case that slipped
    # through inlier count and reprojection error in practice.
    collapsed = np.array([[0.05, 0, 0.8], [0, 0.05, 0.5], [0, 0, 1.0]])
    ok, reason = index._is_plausible(collapsed)
    assert not ok and "collapsed" in reason

    mirrored = np.array([[-1.0, 0, 1.0], [0, 1.0, 0], [0, 0, 1.0]])
    ok, reason = index._is_plausible(mirrored)
    assert not ok and "mirrored" in reason

    exploded = np.array([[50.0, 0, 0], [0, 50.0, 0], [0, 0, 1.0]])
    ok, _ = index._is_plausible(exploded)
    assert not ok

    identity = np.eye(3)
    ok, reason = index._is_plausible(identity)
    assert ok and reason is None

    # A plain translation and a modest zoom are both fine.
    assert index._is_plausible(
        np.array([[1.0, 0, 0.1], [0, 1.0, -0.05], [0, 0, 1.0]])
    )[0]
    assert index._is_plausible(
        np.array([[0.5, 0, 0.25], [0, 0.5, 0.25], [0, 0, 1.0]])
    )[0]


def test_full_view_covers_the_whole_place(world):
    scene = textured_scene(1)
    now = T0
    match = world.begin_visit(scene, now=now)
    world.observe_tracks([track(1, "chair", 0.3, 0.5)], now=now)
    summary = world.end_visit(now=now)
    # Identity transform on a new place: the whole canonical square is seen.
    assert summary["coverage"] == pytest.approx(1.0, abs=0.05)
    assert summary["coverage_established"] is True


def test_partial_scan_does_not_remove_the_unseen_half(world):
    """Gate 5. The single most important behaviour in this module."""
    pano = panorama(2)
    canonical = view(pano, 200)          # the place as first seen
    panned = view(pano, 400)             # camera turned right; left edge lost

    # Anchors are in canonical coordinates: left object near the edge that the
    # panned view will no longer cover, right object well inside it.
    left = track(1, "chair", 0.15, 0.50)
    right = track(2, "laptop", 0.85, 0.50)
    establish(world, canonical, [left, right])

    place_id = 1
    before = {i["class"]: i for i in world.snapshot(place_id, now=T0 + 2 * DAY)}
    assert set(before) == {"chair", "laptop"}

    now = T0 + 3 * DAY
    match = world.begin_visit(panned, now=now)
    assert match.matched and not match.is_new, "panned view is the same place"
    assert match.homography is not None, "need a warp to compute coverage"

    # The laptop sat at canonical 0.85; after a 200 px pan it appears further
    # left in the live frame.
    for _ in range(4):
        world.observe_tracks(
            [track(2, "laptop", 0.55, 0.50)], now=now, match=match
        )
    summary = world.end_visit(now=now)

    assert summary["coverage"] < 0.85, "part of the place was out of frame"

    obs = world.visit_observations(summary["visit_id"])
    by_class = {o["class"]: o["status"] for o in obs.values()}
    assert by_class["laptop"] == ObservationStatus.SEEN
    # The whole point: never looked at, therefore unknown -- not "missing".
    assert by_class["chair"] == ObservationStatus.NOT_OBSERVED

    # The invariant is that *no evidence was appended* -- not that belief is
    # unchanged. Belief always decays with elapsed time under the survival
    # prior, correctly and independently of whether we looked.
    after = {i["class"]: i for i in world.snapshot(place_id, now=now)}
    assert after["chair"]["opportunities"] == before["chair"]["opportunities"], (
        "an unchecked object must gain no observation"
    )
    assert after["laptop"]["opportunities"] > before["laptop"]["opportunities"]
    assert world.changes(place_id, now=now)["removed"] == []


def test_occluded_object_is_uncertain_not_missing(world):
    """Gate 6. Someone standing in front of the laptop is not a removal."""
    scene = textured_scene(3)
    laptop = track(2, "laptop", 0.70, 0.50)
    establish(world, scene, [track(1, "chair", 0.20, 0.50), laptop])

    place_id = 1
    before = {i["class"]: i for i in world.snapshot(place_id, now=T0 + 2 * DAY)}

    now = T0 + 3 * DAY
    match = world.begin_visit(scene, now=now)
    for _ in range(4):
        world.observe_tracks(
            [
                track(1, "chair", 0.20, 0.50),
                # A person occupying the laptop's position.
                track(9, "person", 0.70, 0.50, half=0.18),
            ],
            now=now, match=match,
        )
    summary = world.end_visit(now=now)

    obs = world.visit_observations(summary["visit_id"])
    by_class = {o["class"]: o["status"] for o in obs.values()}
    assert by_class["laptop"] == ObservationStatus.OCCLUDED_UNCERTAIN
    assert summary["occluded"] >= 1

    after = {i["class"]: i for i in world.snapshot(place_id, now=now)}
    assert after["laptop"]["opportunities"] == before["laptop"]["opportunities"], (
        "being blocked from view is not evidence of anything"
    )
    assert world.changes(place_id, now=now)["removed"] == []


def test_checked_missing_still_reports_removal(world):
    """The control: with full coverage and nothing in the way, a genuinely
    absent object must still be reported. Coverage must not make the system
    unable to conclude anything."""
    scene = textured_scene(4)
    chair = track(1, "chair", 0.20, 0.50)
    laptop = track(2, "laptop", 0.70, 0.50)
    establish(world, scene, [chair, laptop])

    for day in (3, 4, 5):
        now = T0 + day * DAY
        match = world.begin_visit(scene, now=now)
        for _ in range(4):
            world.observe_tracks([chair], now=now, match=match)
        summary = world.end_visit(now=now)

    obs = world.visit_observations(summary["visit_id"])
    by_class = {o["class"]: o["status"] for o in obs.values()}
    assert by_class["laptop"] == ObservationStatus.CHECKED_MISSING

    now = T0 + 5 * DAY
    assert [c["class"] for c in world.changes(1, now=now)["removed"]] == ["laptop"]


def test_one_frame_false_detection_never_becomes_memory(world):
    """A detector hallucination must not turn into 'there used to be a
    backpack here'."""
    scene = textured_scene(5)
    now = T0
    match = world.begin_visit(scene, now=now)
    for i in range(6):
        objects = [track(1, "chair", 0.30, 0.50)]
        if i == 2:  # a single spurious frame
            objects.append(track(99, "backpack", 0.75, 0.30))
        world.observe_tracks(objects, now=now, match=match)
    summary = world.end_visit(now=now)

    assert summary["discarded_tentative"] >= 1
    assert "backpack" not in {i["class"] for i in world.snapshot(1, now=now)}
    rows = world.conn.execute(
        "SELECT COUNT(*) FROM object_instances WHERE cls = 'backpack'"
    ).fetchone()[0]
    assert rows == 0, "the spurious instance must be deleted, not just hidden"


def test_short_visit_produces_no_absence_evidence(world):
    """A five-second accidental recording must not mark the room missing."""
    scene = textured_scene(6)
    chair = track(1, "chair", 0.20, 0.50)
    laptop = track(2, "laptop", 0.70, 0.50)
    establish(world, scene, [chair, laptop])
    before = {i["class"]: i for i in world.snapshot(1, now=T0 + 2 * DAY)}

    # A visit with essentially no coverage: the transform never became
    # trustworthy, so nothing was established.
    now = T0 + 3 * DAY
    world.begin_visit(np.full((480, 640, 3), 30, np.uint8), now=now)
    summary = world.end_visit(now=now)

    after = {i["class"]: i for i in world.snapshot(1, now=now)}
    for cls, prev in before.items():
        assert after[cls]["opportunities"] == prev["opportunities"], (
            f"{cls} gained evidence from a visit that established nothing"
        )
    assert summary.get("checked_missing", 0) == 0


def test_end_visit_is_idempotent(world):
    """Explicit stop and socket disconnect can both fire."""
    scene = textured_scene(7)
    chair = track(1, "chair", 0.30, 0.50)
    establish(world, scene, [chair])

    now = T0 + 3 * DAY
    match = world.begin_visit(scene, now=now)
    for _ in range(4):
        world.observe_tracks([chair], now=now, match=match)
    first = world.end_visit(now=now)
    second = world.end_visit(now=now)

    assert first["visit_id"] is not None
    assert second == {}, "a second close must be a no-op"

    inst = world.snapshot(1, now=now)[0]
    assert inst["opportunities"] == 4, "evidence must not be double-counted"
