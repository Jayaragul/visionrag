"""Demonstrate persistent world memory over a simulated week.

Runs a fixed scenario -- an office desk visited daily -- and prints what the
system comes to believe about each object. No camera or model needed; the
point is the reasoning layer, so tracks are supplied directly.

    python scripts/demo_world.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionrag.memory.persistence import SECONDS_PER_DAY  # noqa: E402
from visionrag.memory.places import GeoFix, PlaceIndex  # noqa: E402
from visionrag.memory.world import WorldMemory  # noqa: E402
from visionrag.types import Track, TrackState  # noqa: E402

DAY = SECONDS_PER_DAY
T0 = 1_760_000_000.0


def track(tid: int, cls: str, cx: float, cy: float) -> Track:
    h = 0.05
    return Track(
        track_id=tid, cls=cls, first_seen_ms=0, last_seen_ms=1000,
        states=[TrackState(0, 1000, (cx - h, cy - h, cx + h, cy + h), 0.9)],
        confirmed=True,
    )


def desk_view() -> np.ndarray:
    rng = np.random.default_rng(1234)
    img = np.full((240, 320, 3), 50, dtype=np.uint8)
    for _ in range(14):
        x, y = int(rng.integers(0, 260)), int(rng.integers(0, 180))
        c = tuple(int(v) for v in rng.integers(40, 255, 3))
        cv2.rectangle(img, (x, y), (x + 55, y + 55), c, -1)
    return img


# day -> objects present. The monitor and desk are fixtures; the laptop is
# taken away after day 3; a person is present most days; a coffee cup shows up
# once. Day 5 has a missed laptop-free detection of the monitor to show that a
# single miss is tolerated.
SCHEDULE = {
    0: ["tv", "dining table", "laptop", "person"],
    1: ["tv", "dining table", "laptop", "person", "cup"],
    2: ["tv", "dining table", "laptop", "person"],
    3: ["tv", "dining table", "person"],
    4: ["tv", "dining table", "person"],
    5: ["dining table", "person"],          # monitor missed by the detector
    6: ["tv", "dining table"],
}

ANCHOR = {
    "tv": (0.50, 0.30),
    "dining table": (0.50, 0.75),
    "laptop": (0.35, 0.55),
    "person": (0.70, 0.55),
    "cup": (0.20, 0.50),
}


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / "world.db"
    view = desk_view()
    here = GeoFix(lat=12.9716, lon=77.5946, accuracy_m=9.0)

    with WorldMemory(tmp, index=PlaceIndex(verify_geometry=False)) as world:
        print("simulating 7 daily visits to one desk\n")
        for day, classes in SCHEDULE.items():
            now = T0 + day * DAY
            match = world.begin_visit(view, geo=here, now=now)
            place = match.place
            tracks = [
                track(i, c, *ANCHOR[c]) for i, c in enumerate(classes, start=1)
            ]
            # Only the person is ever observed moving within a visit.
            moving = {i for i, c in enumerate(classes, start=1) if c == "person"}
            # Several analysed frames per visit: one sighting is deliberately
            # not enough for an object to enter permanent memory.
            for _ in range(5):
                world.observe_tracks(tracks, now=now, moving_ids=moving)
            summary = world.end_visit(now=now)
            tag = "  (new place)" if match.is_new else ""
            print(
                f"  day {day}: saw {len(classes)} object(s); "
                f"{summary['checked_missing']} checked-missing, "
                f"{summary['not_observed']} not checked{tag}"
            )

        now = T0 + 6 * DAY
        place_id = place.place_id

        print("\nwhat the system believes is at this place:\n")
        header = f"  {'object':<15}{'kind':<16}{'tier':<14}{'still there':>12}{'seen':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in world.snapshot(place_id, now=now):
            seen = f"{i['times_seen']}/{i['opportunities']}"
            print(
                f"  {i['class']:<15}{i['semantic_kind']:<16}{i['tier']:<14}"
                f"{i['persistence']:>12.3f}{seen:>8}"
            )

        changes = world.changes(place_id, now=now)
        print("\nchange report (dynamic objects excluded):\n")
        for label, key in (("gone", "removed"), ("new", "added"),
                           ("unchanged", "unchanged")):
            names = [c["class"] for c in changes[key]]
            print(f"  {label:<11}{', '.join(names) if names else '-'}")

        print("\n  where did I last see the laptop?")
        for r in world.find("laptop", now=now):
            print(
                f"    place {r['place_id']} | still there {r['still_there']:.3f}"
                f" | tier {r['tier']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
