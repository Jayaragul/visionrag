"""Tests for persistence estimation and place recognition.

The persistence filter is the piece that decides what the system believes is
permanent, so its behaviour is pinned down here rather than eyeballed on
video. Every case is analytic and runs in milliseconds.
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
    ObservationStatus,
    PersistenceFilter,
    SemanticKind,
    Tier,
    classify,
    is_evidence,
    lifetime_prior,
    semantic_kind,
)
from visionrag.memory.places import GeoFix, PlaceIndex  # noqa: E402

DAY = SECONDS_PER_DAY


def _filter(lifetime_days: float = 14.0) -> PersistenceFilter:
    return PersistenceFilter(lifetime_s=lifetime_days * DAY, p_miss=0.3, p_false=0.02)


# -- persistence filter -------------------------------------------------
def test_no_evidence_falls_back_to_prior():
    f = _filter(14.0)
    assert f.probability(0.0) == 1.0
    # With no observations the answer is exactly the survival prior.
    assert abs(f.probability(14 * DAY) - np.exp(-1.0)) < 1e-9


def test_repeated_sightings_keep_persistence_high():
    f = _filter(14.0)
    for day in range(6):
        f.observe(day * DAY, detected=True)
    assert f.probability(5 * DAY) > 0.95


def test_repeated_absence_drives_persistence_down():
    f = _filter(14.0)
    f.observe(0.0, detected=True)
    f.observe(1 * DAY, detected=True)
    for day in range(2, 8):
        f.observe(day * DAY, detected=False)
    # Six consecutive misses is far more likely to mean "removed" than
    # "the detector failed six times in a row".
    assert f.probability(7 * DAY) < 0.05


def test_single_miss_is_tolerated():
    """A detector with a 30% miss rate will drop objects that are present.
    One absence among many sightings must not be read as removal."""
    f = _filter(14.0)
    for day in range(8):
        f.observe(day * DAY, detected=(day != 4))
    assert f.probability(7 * DAY) > 0.80


def test_class_prior_separates_permanent_from_transient():
    """Same observation history, different priors: a wall should outlast a cup
    once neither has been seen for a while."""
    history = [(0.0, True), (1 * DAY, True), (2 * DAY, True)]
    wall = PersistenceFilter(lifetime_s=lifetime_prior("wall"))
    cup = PersistenceFilter(lifetime_s=lifetime_prior("cup"))
    for t, seen in history:
        wall.observe(t, seen)
        cup.observe(t, seen)
    later = 10 * DAY
    assert wall.probability(later) > 0.99
    assert cup.probability(later) < wall.probability(later)


def test_persistence_decays_with_time_since_last_look():
    f = _filter(7.0)
    f.observe(0.0, detected=True)
    f.observe(1 * DAY, detected=True)
    near = f.probability(2 * DAY)
    far = f.probability(60 * DAY)
    assert near > far
    assert far < 0.05  # long past the expected lifetime


def test_probability_stays_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = _filter(float(rng.uniform(0.5, 100)))
        for i in range(int(rng.integers(0, 12))):
            f.observe(i * DAY, detected=bool(rng.integers(0, 2)))
        p = f.probability(float(rng.uniform(0, 200)) * DAY)
        assert 0.0 <= p <= 1.0


# -- tier classification ------------------------------------------------
def test_motion_outranks_presence_statistics():
    """A person reliably present across visits is still dynamic. Direct
    evidence of motion must beat presence statistics."""
    assert classify(n_visits=10, observed_moving=True, hit_rate=1.0) == Tier.DYNAMIC


def test_tiers_come_from_consistency_not_current_presence():
    # Seen on every visit -> fixtures.
    assert classify(n_visits=5, observed_moving=False, hit_rate=1.0) == Tier.STATIC
    # A single detector miss must not demote a fixture.
    assert classify(n_visits=7, observed_moving=False, hit_rate=6 / 7) == Tier.STATIC
    # Comes and goes -> furniture.
    assert (
        classify(n_visits=5, observed_moving=False, hit_rate=0.4) == Tier.SEMI_STATIC
    )
    # Currently absent, but it is still the kind of thing that sits on a desk.
    # Filing this under "dynamic" would hide it from change reporting.
    assert (
        classify(n_visits=6, observed_moving=False, hit_rate=0.33) == Tier.SEMI_STATIC
    )


def test_stationary_person_is_never_world_structure():
    """The bug this taxonomy exists to prevent.

    Someone who sits at their desk without moving scores a perfect hit rate
    and no motion evidence. Without a semantic constraint they are classified
    as building structure. Motion is evidence *for* dynamic; its absence is
    not evidence for furniture.
    """
    assert (
        classify(
            n_visits=10,
            hit_rate=1.0,
            kind=SemanticKind.PERSON,
            observed_moving=False,
        )
        == Tier.DYNAMIC
    )


def test_semantic_kind_caps_reachable_tiers():
    # A backpack seen every single visit is reliable, but never a fixture.
    assert (
        classify(n_visits=10, hit_rate=1.0, kind=SemanticKind.MOVABLE)
        == Tier.SEMI_STATIC
    )
    # Built-in appliances may be static.
    assert (
        classify(n_visits=10, hit_rate=1.0, kind=SemanticKind.FIXTURE)
        == Tier.STATIC
    )
    # So may large furniture.
    assert (
        classify(n_visits=10, hit_rate=1.0, kind=SemanticKind.FURNITURE)
        == Tier.STATIC
    )
    # A car parked in the same spot daily is semi-static, never structure.
    assert (
        classify(n_visits=10, hit_rate=1.0, kind=SemanticKind.VEHICLE)
        == Tier.SEMI_STATIC
    )


def test_class_to_kind_mapping():
    assert semantic_kind("person") == SemanticKind.PERSON
    assert semantic_kind("chair") == SemanticKind.FURNITURE
    assert semantic_kind("refrigerator") == SemanticKind.FIXTURE
    assert semantic_kind("backpack") == SemanticKind.MOVABLE
    assert semantic_kind("car") == SemanticKind.VEHICLE
    assert semantic_kind("dog") == SemanticKind.ANIMAL
    # An unmapped class must not be silently treated as furniture.
    assert semantic_kind("flux capacitor") == SemanticKind.UNKNOWN


def test_only_evidential_statuses_count():
    """`not_observed` and `occluded` must never become evidence of removal."""
    assert is_evidence(ObservationStatus.SEEN)
    assert is_evidence(ObservationStatus.CHECKED_MISSING)
    assert not is_evidence(ObservationStatus.NOT_OBSERVED)
    assert not is_evidence(ObservationStatus.OCCLUDED_UNCERTAIN)


def test_single_visit_is_unknown():
    """One visit cannot separate 'always here' from 'here right now'."""
    assert classify(n_visits=1, observed_moving=False, hit_rate=1.0) == Tier.UNKNOWN


# -- place recognition --------------------------------------------------
def _scene(seed: int, size=(320, 240)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    for _ in range(14):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        colour = tuple(int(c) for c in rng.integers(40, 255, 3))
        cv2.rectangle(img, (x, y), (x + 55, y + 55), colour, -1)
    return img


def test_same_view_is_recognised_as_one_place():
    idx = PlaceIndex(verify_geometry=False)
    scene = _scene(1)
    a = idx.observe(scene, ts_ms=0)
    b = idx.observe(scene.copy(), ts_ms=1000)
    assert a.is_new is True and b.is_new is False
    assert a.place.place_id == b.place.place_id
    assert len(idx.places) == 1


def test_different_views_create_different_places():
    idx = PlaceIndex(verify_geometry=False)
    idx.observe(_scene(1), ts_ms=0)
    idx.observe(_scene(99), ts_ms=1000)
    assert len(idx.places) == 2


def test_gps_separates_lookalike_places():
    """Perceptual aliasing: identical-looking views far apart must not merge.
    This is the failure mode GPS gating exists to prevent."""
    idx = PlaceIndex(verify_geometry=False, gps_gate_m=60.0)
    scene = _scene(7)
    here = GeoFix(lat=12.9716, lon=77.5946, accuracy_m=8.0)
    # ~1.1 km north -- far outside the gate.
    far = GeoFix(lat=12.9816, lon=77.5946, accuracy_m=8.0)

    p1 = idx.observe(scene, ts_ms=0, geo=here).place
    p2 = idx.observe(scene.copy(), ts_ms=1000, geo=far).place
    assert p1.place_id != p2.place_id

    # Returning to the first location recognises the original place.
    m3 = idx.observe(scene.copy(), ts_ms=2000, geo=here)
    p3, is_new = m3.place, m3.is_new
    assert is_new is False
    assert p3.place_id == p1.place_id


def test_imprecise_gps_does_not_veto_appearance():
    """Indoors the browser reports huge accuracy radii. Such a fix must be
    ignored rather than allowed to reject a correct appearance match."""
    idx = PlaceIndex(verify_geometry=False, gps_trust_accuracy_m=50.0)
    scene = _scene(3)
    good = GeoFix(lat=12.9716, lon=77.5946, accuracy_m=10.0)
    indoors = GeoFix(lat=13.5, lon=78.5, accuracy_m=2000.0)  # nonsense, unusable

    p1 = idx.observe(scene, ts_ms=0, geo=good).place
    m2 = idx.observe(scene.copy(), ts_ms=1000, geo=indoors)
    p2, is_new = m2.place, m2.is_new
    assert is_new is False
    assert p2.place_id == p1.place_id


def test_geometric_verification_prevents_false_merges():
    """Histogram similarity has no notion of structure, so distinct scenes can
    score above a permissive threshold. ORB + homography rejects those.

    Measured: at threshold 0.72 without verification, 3 of 11 distinct scenes
    merged. With verification, none did — and no revisit was missed.
    """
    seeds = [1, 11, 21, 31, 41, 51, 52, 61, 62, 71, 99]

    loose = PlaceIndex(match_thresh=0.72, verify_geometry=False)
    for i, s in enumerate(seeds):
        loose.observe(_scene(s), ts_ms=i * 1000)
    assert len(loose.places) < len(seeds), "expected false merges without verification"

    verified = PlaceIndex(match_thresh=0.72, verify_geometry=True)
    for i, s in enumerate(seeds):
        verified.observe(_scene(s), ts_ms=i * 1000)
    assert len(verified.places) == len(seeds)

    # Verification must not cause the opposite failure: every revisit should
    # still be recognised rather than spawning a duplicate place.
    for i, s in enumerate(seeds):
        is_new = verified.observe(_scene(s), ts_ms=(100 + i) * 1000).is_new
        assert is_new is False


def test_homography_maps_live_to_canonical():
    """Pin the transform direction.

    `H` must map *live* coordinates into the place's *canonical* frame. Getting
    this backwards still produces a valid-looking homography, so nothing fails
    loudly -- anchors just drift in a way that looks like random noise.

    The scene is translated by a known amount, which is what a sideways step
    approximates. An object that did not move must land back on its original
    canonical position after warping.
    """
    idx = PlaceIndex(verify_geometry=True)
    scene = _scene(5, size=(640, 480))
    first = idx.observe(scene, ts_ms=0)
    assert first.is_new
    # A new place is its own canonical frame: identity, and trusted.
    assert first.homography is not None
    assert first.to_canonical(0.62, 0.55) == pytest.approx((0.62, 0.55), abs=1e-6)

    # Shift the whole scene right by 64 px of 640 = 0.1 normalised.
    shift_px = 64
    shifted = np.roll(scene, shift_px, axis=1)
    match = idx.observe(shifted, ts_ms=1000)

    assert match.matched and not match.is_new, "translated view is the same place"
    assert match.homography is not None, "translation should be a trustworthy warp"

    # A stationary object appears 0.1 further right in the live frame; warping
    # must put it back where it was.
    live_x, live_y = 0.62 + 0.1, 0.55
    canonical = match.to_canonical(live_x, live_y)
    assert canonical is not None
    assert canonical[0] == pytest.approx(0.62, abs=0.02)
    assert canonical[1] == pytest.approx(0.55, abs=0.02)


def test_untrusted_homography_is_withheld():
    """A weak fit must yield no transform at all rather than a bad one.

    A confidently wrong homography is worse than none: it moves anchors
    decisively to the wrong place, which reads downstream as objects having
    moved.
    """
    idx = PlaceIndex(verify_geometry=True, min_warp_inliers=10_000)
    scene = _scene(6, size=(640, 480))
    idx.observe(scene, ts_ms=0)
    match = idx.observe(scene.copy(), ts_ms=1000)
    assert match.matched, "still recognised as the same place"
    assert match.verification.inliers > 0
    assert match.homography is None, "unreachable inlier bar must withhold the warp"
    assert match.to_canonical(0.5, 0.5) is None


def test_frame_corners_give_coverage_polygon():
    idx = PlaceIndex(verify_geometry=True)
    scene = _scene(8, size=(640, 480))
    match = idx.observe(scene, ts_ms=0)
    corners = match.frame_corners_canonical()
    assert corners is not None and corners.shape == (4, 2)
    # Identity transform on a new place: the canonical view is the unit square.
    assert corners.min() == pytest.approx(0.0, abs=1e-6)
    assert corners.max() == pytest.approx(1.0, abs=1e-6)


def test_works_with_no_gps_at_all():
    idx = PlaceIndex(verify_geometry=False)
    scene = _scene(5)
    p1 = idx.observe(scene, ts_ms=0, geo=None).place
    m2 = idx.observe(scene.copy(), ts_ms=500, geo=None)
    p2, is_new = m2.place, m2.is_new
    assert is_new is False and p1.place_id == p2.place_id
