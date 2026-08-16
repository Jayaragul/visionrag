"""Cross-session world memory: does it learn what is permanent?

Simulates repeated visits to one place over several days and checks that the
system separates furniture from clutter from people, and notices when
something is taken away.
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
    SemanticKind,
    Tier,
)
from visionrag.memory.places import GeoFix, PlaceIndex  # noqa: E402
from visionrag.memory.world import WorldMemory  # noqa: E402
from visionrag.types import Track, TrackState  # noqa: E402

DAY = SECONDS_PER_DAY
T0 = 1_760_000_000.0  # fixed epoch so tests are deterministic


def track(track_id: int, cls: str, cx: float, cy: float) -> Track:
    half = 0.05
    box = (cx - half, cy - half, cx + half, cy + half)
    return Track(
        track_id=track_id,
        cls=cls,
        first_seen_ms=0,
        last_seen_ms=1000,
        states=[TrackState(frame_id=0, ts_ms=1000, box=box, score=0.9)],
        confirmed=True,
    )


def scene(seed: int, size=(320, 240)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    for _ in range(14):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        colour = tuple(int(c) for c in rng.integers(40, 255, 3))
        cv2.rectangle(img, (x, y), (x + 55, y + 55), colour, -1)
    return img


@pytest.fixture()
def world(tmp_path):
    with WorldMemory(
        tmp_path / "world.db", index=PlaceIndex(verify_geometry=False)
    ) as w:
        yield w


def visit(world, view, day, objects, moving_ids=None, geo=None, frames=4):
    """One visit. `frames` analysed frames, because a single frame is not
    enough for an instance to clear the confirmation bar -- by design."""
    now = T0 + day * DAY
    match = world.begin_visit(view, geo=geo, now=now)
    for _ in range(frames):
        world.observe_tracks(objects, now=now, moving_ids=moving_ids or set())
    summary = world.end_visit(now=now)
    return match.place, summary


def test_learns_permanent_versus_temporary(world):
    view = scene(11)
    chair, laptop = (0.30, 0.60), (0.70, 0.45)

    # Six daily visits. The chair is always present. The laptop is present
    # for the first three days, then taken away.
    for day in range(6):
        objects = [track(1, "chair", *chair)]
        if day < 3:
            objects.append(track(2, "laptop", *laptop))
        visit(world, view, day, objects)

    place_id = 1
    snap = {i["class"]: i for i in world.snapshot(place_id, now=T0 + 5 * DAY)}

    # The chair was seen at every opportunity, so belief should stay high.
    assert snap["chair"]["persistence"] > 0.9
    assert snap["chair"]["times_seen"] == 6

    # The laptop was missed three times running -- far likelier removed than
    # missed three times by a detector.
    assert snap["laptop"]["persistence"] < 0.2

    changes = world.changes(place_id, now=T0 + 5 * DAY)
    assert [c["class"] for c in changes["removed"]] == ["laptop"]
    assert "chair" in [c["class"] for c in changes["unchanged"]]


def test_people_are_never_treated_as_furniture(world):
    """Someone at their desk every single day is still dynamic. Presence
    statistics must not promote a person to permanent scenery."""
    view = scene(21)
    for day in range(6):
        visit(
            world,
            view,
            day,
            [track(1, "chair", 0.3, 0.6), track(9, "person", 0.5, 0.5)],
            moving_ids={9},
        )

    snap = {i["class"]: i for i in world.snapshot(1, now=T0 + 5 * DAY)}
    assert snap["person"]["tier"] == Tier.DYNAMIC
    assert snap["chair"]["tier"] in (Tier.STATIC, Tier.SEMI_STATIC)

    # A person walking through is not a change to the world.
    changes = world.changes(1, now=T0 + 5 * DAY)
    for bucket in changes.values():
        assert "person" not in [c["class"] for c in bucket]


def test_motionless_person_still_not_furniture(world):
    """The harder version of the previous test: no motion evidence at all.

    Someone sitting still at their desk every day is present on every visit
    and never observed moving. Nothing but semantic kind can stop them being
    classified as building structure.
    """
    view = scene(22)
    for day in range(6):
        visit(
            world,
            view,
            day,
            [track(1, "chair", 0.3, 0.6), track(9, "person", 0.5, 0.5)],
            moving_ids=set(),  # deliberately never marked as moving
        )

    snap = {i["class"]: i for i in world.snapshot(1, now=T0 + 5 * DAY)}
    assert snap["person"]["tier"] == Tier.DYNAMIC
    assert snap["person"]["semantic_kind"] == SemanticKind.PERSON

    changes = world.changes(1, now=T0 + 5 * DAY)
    for bucket in changes.values():
        assert "person" not in [c["class"] for c in bucket]


def test_movable_object_never_becomes_a_fixture(world):
    """A backpack left on the desk every single day is reliable, not built-in."""
    view = scene(23)
    for day in range(6):
        visit(world, view, day, [track(1, "backpack", 0.4, 0.5)])

    snap = {i["class"]: i for i in world.snapshot(1, now=T0 + 5 * DAY)}
    assert snap["backpack"]["tier"] == Tier.SEMI_STATIC
    assert snap["backpack"]["semantic_kind"] == SemanticKind.MOVABLE


def test_single_miss_does_not_declare_removal(world):
    """The detector misses things. One absence among many sightings must not
    be reported as an object being taken away."""
    view = scene(31)
    for day in range(8):
        objects = [track(1, "chair", 0.3, 0.6)]
        if day != 4:  # one missed detection
            objects.append(track(2, "laptop", 0.7, 0.45))
        visit(world, view, day, objects)

    changes = world.changes(1, now=T0 + 7 * DAY)
    assert [c["class"] for c in changes["removed"]] == []
    assert "laptop" in [c["class"] for c in changes["unchanged"]]


def test_returning_to_a_place_is_recognised(world):
    view = scene(41)
    here = GeoFix(lat=12.9716, lon=77.5946, accuracy_m=9.0)
    p1, _ = visit(world, view, 0, [track(1, "chair", 0.3, 0.6)], geo=here)
    p2, _ = visit(world, view, 1, [track(1, "chair", 0.3, 0.6)], geo=here)
    assert p1.place_id == p2.place_id
    assert world.index.places[p1.place_id].n_visits == 2


def test_different_rooms_stay_separate(world):
    """Objects must not leak between places -- a chair in the kitchen is not
    evidence about the office."""
    kitchen, office = scene(51), scene(52)
    visit(world, kitchen, 0, [track(1, "chair", 0.3, 0.6)])
    visit(world, office, 0, [track(1, "laptop", 0.5, 0.5)])

    assert len(world.index.places) == 2
    classes_by_place = {
        pid: {i["class"] for i in world.snapshot(pid)}
        for pid in world.index.places
    }
    assert {"chair"} in classes_by_place.values()
    assert {"laptop"} in classes_by_place.values()


def test_find_ranks_by_belief_not_recency(world):
    """'Where is the backpack?' should prefer the place it is probably still
    at, not simply the last place it was seen.

    Timescale matters here: a backpack's prior lifetime is ~12 hours, so over
    a span of days belief in *any* backpack correctly decays to zero and there
    is nothing left to rank. The question is only meaningful within the
    object's own lifetime, so this works in hours.
    """
    hour = 3600.0
    desk, hallway = scene(61), scene(62)

    # Established at the desk across five hourly visits.
    for h in range(5):
        visit(world, desk, h * hour / DAY, [track(1, "backpack", 0.4, 0.5)])
    # Glimpsed once in the hallway, more recently, then absent three times.
    visit(world, hallway, 5 * hour / DAY, [track(1, "backpack", 0.5, 0.5)])
    for h in range(6, 9):
        visit(world, hallway, h * hour / DAY, [])

    results = world.find("backpack", now=T0 + 8 * hour)
    assert len(results) == 2
    # The hallway sighting is more recent, but it was looked for three times
    # since and not found; the desk is where it probably still is.
    assert results[0]["place_id"] != results[1]["place_id"]
    assert results[0]["still_there"] > results[1]["still_there"]


def test_persistence_survives_reopening_the_database(tmp_path):
    """World memory is worthless if it forgets on restart."""
    view = scene(71)
    db = tmp_path / "world.db"
    for day in range(3):
        with WorldMemory(db, index=PlaceIndex(verify_geometry=False)) as w:
            visit(w, view, day, [track(1, "chair", 0.3, 0.6)])

    with WorldMemory(db, index=PlaceIndex(verify_geometry=False)) as w:
        assert len(w.index.places) == 1
        snap = w.snapshot(1, now=T0 + 2 * DAY)
        assert len(snap) == 1
        assert snap[0]["class"] == "chair"
        assert snap[0]["times_seen"] == 3
