"""Object persistence estimation.

Answers "is this thing still there?" from a history of detections *and*
missed detections, rather than assuming permanence from a class label.

Implements the persistence filter of Rosen, Mason & Leonard, "Towards Lifelong
Feature-Based Mapping in Semi-Static Environments" (ICRA 2016), adapted from
map landmarks to object instances anchored at topological places.

Why not just count sightings
----------------------------
A naive hit-rate cannot tell "the chair was removed" from "the detector missed
the chair", and it has no notion of time -- an object seen 10 times last year
scores the same as one seen 10 times today. The filter handles both: detector
fallibility enters through explicit miss/false-alarm rates, and time enters
through a survival prior.

The model
---------
An object's lifetime ``T`` is drawn from an exponential survival prior with
``S(t) = exp(-lambda * t)``. At visit times ``t_1 < ... < t_N`` we get binary
observations ``y_i``. The detector is characterised by:

    P(y = 1 | object present) = 1 - P_M     (P_M = miss rate)
    P(y = 1 | object absent)  = P_F         (P_F = false-alarm rate)

To get the posterior we partition on *which interval the object died in*. If
``T`` falls in ``[t_k, t_{k+1})`` then observations up to ``k`` saw a present
object and those after saw an absent one, so that partition's likelihood is a
product of two runs. Summing the partitions weighted by their prior mass gives
the normaliser, and the surviving branch gives the answer:

    P(alive at t) = S(t) * L_N / Z

Everything is O(N) per query and runs in a few microseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Expected lifetime in seconds, by coarse object category. These are *priors*:
# they decide what the estimate looks like before evidence accumulates, and
# are progressively overridden by observation. Values are deliberately coarse
# -- an order of magnitude is the meaningful unit here, not a precise number.
SECONDS_PER_DAY = 86_400.0

CLASS_LIFETIME_PRIOR: dict[str, float] = {
    # Built structure: effectively permanent on any horizon we operate over.
    "wall": 3650 * SECONDS_PER_DAY,
    "door": 3650 * SECONDS_PER_DAY,
    "window": 3650 * SECONDS_PER_DAY,
    "refrigerator": 1000 * SECONDS_PER_DAY,
    "oven": 1000 * SECONDS_PER_DAY,
    "sink": 1000 * SECONDS_PER_DAY,
    "toilet": 1000 * SECONDS_PER_DAY,
    "bed": 365 * SECONDS_PER_DAY,
    "couch": 365 * SECONDS_PER_DAY,
    "dining table": 365 * SECONDS_PER_DAY,
    "tv": 180 * SECONDS_PER_DAY,
    "potted plant": 90 * SECONDS_PER_DAY,
    # Semi-static: stationary in any one visit, moved between visits.
    "chair": 14 * SECONDS_PER_DAY,
    "laptop": 3 * SECONDS_PER_DAY,
    "keyboard": 7 * SECONDS_PER_DAY,
    "mouse": 7 * SECONDS_PER_DAY,
    "book": 7 * SECONDS_PER_DAY,
    "vase": 30 * SECONDS_PER_DAY,
    "bowl": 1 * SECONDS_PER_DAY,
    "cup": 0.25 * SECONDS_PER_DAY,
    "bottle": 1 * SECONDS_PER_DAY,
    "backpack": 0.5 * SECONDS_PER_DAY,
    "handbag": 0.5 * SECONDS_PER_DAY,
    "suitcase": 1 * SECONDS_PER_DAY,
    "cell phone": 0.1 * SECONDS_PER_DAY,
    "bicycle": 2 * SECONDS_PER_DAY,
    # Dynamic: present only while being observed.
    "person": 120.0,
    "cat": 600.0,
    "dog": 600.0,
    "car": 3600.0,
    "truck": 3600.0,
    "bus": 600.0,
}

DEFAULT_LIFETIME = 1.0 * SECONDS_PER_DAY


def lifetime_prior(cls: str) -> float:
    return CLASS_LIFETIME_PRIOR.get(cls, DEFAULT_LIFETIME)


class ObservationStatus:
    """What actually happened when we had a chance to see an object.

    Two states are not enough. "I looked and it was gone" and "I never looked
    over there" are completely different claims, and collapsing them makes the
    system report objects as removed whenever the camera was pointed
    elsewhere. This is the object-level form of the distinction occupancy-grid
    mapping draws between *observed empty* and *unobserved* (Moravec & Elfes,
    1985): unknown is not the same as absent.
    """

    SEEN = "seen"
    CHECKED_MISSING = "checked_missing"        # in view, unoccluded, absent
    OCCLUDED_UNCERTAIN = "occluded_uncertain"  # in view but blocked
    NOT_OBSERVED = "not_observed"              # never looked

    # Only these carry evidence. The other two append *nothing* to the filter,
    # which then decays under the survival prior alone -- exactly the right
    # meaning for "I haven't checked in three days".
    EVIDENTIAL = frozenset({SEEN, CHECKED_MISSING})


def is_evidence(status: str) -> bool:
    return status in ObservationStatus.EVIDENTIAL


class InstanceState:
    """Lifecycle of a remembered object.

    A single false detection must never become "there used to be a backpack
    here", so nothing enters permanent memory until repeated evidence confirms
    it. This is the track state machine in `ingest/track.py` (`min_hits` ->
    `confirmed`) lifted from frames to visits.

        TENTATIVE --repeated evidence--> CONFIRMED --long absence--> DORMANT
            |                                                          |
            +-- vanishes --> discarded              strong evidence --> REMOVED
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DORMANT = "dormant"
    REMOVED = "removed"


@dataclass
class Observation:
    """One opportunity to see an object, and whether it was seen.

    The absent case matters as much as the present one -- an object's
    persistence should fall when you look and it is not there. A store of
    positive sightings alone can never conclude that something was removed.
    """

    t_s: float  # seconds since the object was first seen
    detected: bool


@dataclass
class PersistenceFilter:
    """Posterior belief that an object still exists.

    `p_miss` and `p_false` characterise the detector at the operating
    threshold. They should come from a measured PR curve, not intuition:
    over-optimistic values make the filter too eager to declare removals.
    """

    lifetime_s: float = DEFAULT_LIFETIME
    p_miss: float = 0.30
    p_false: float = 0.02
    observations: list[Observation] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return 1.0 / max(1e-9, self.lifetime_s)

    def survival(self, t_s: float) -> float:
        """S(t) = P(T > t) under the exponential prior."""
        return math.exp(-self.rate * max(0.0, t_s))

    def observe(self, t_s: float, detected: bool) -> None:
        self.observations.append(Observation(t_s, detected))

    # -- inference --------------------------------------------------------
    def _log_present(self, detected: bool) -> float:
        p = (1.0 - self.p_miss) if detected else self.p_miss
        return math.log(max(1e-12, p))

    def _log_absent(self, detected: bool) -> float:
        p = self.p_false if detected else (1.0 - self.p_false)
        return math.log(max(1e-12, p))

    def probability(self, t_now_s: float) -> float:
        """P(object still exists at `t_now_s`) given all observations."""
        obs = sorted(self.observations, key=lambda o: o.t_s)
        if not obs:
            # No evidence at all: the prior is the whole answer.
            return self.survival(t_now_s)

        n = len(obs)
        # suffix_absent[k] = log-likelihood of observations k..n-1 assuming the
        # object was already gone when they were made.
        suffix_absent = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_absent[i] = suffix_absent[i + 1] + self._log_absent(obs[i].detected)

        times = [o.t_s for o in obs]
        # Partition k: the object died in [t_k, t_{k+1}), so observations
        # 0..k-1 saw it present and k..n-1 saw it absent.
        terms: list[float] = []
        prefix_present = 0.0
        for k in range(n):
            t_lo = times[k - 1] if k > 0 else 0.0
            t_hi = times[k]
            mass = self.survival(t_lo) - self.survival(t_hi)
            if mass > 0.0:
                terms.append(math.log(mass) + prefix_present + suffix_absent[k])
            prefix_present += self._log_present(obs[k].detected)

        # Final partition: died after the last observation, or not at all.
        # Its prior mass is S(t_N); the surviving branch carries S(t_now).
        log_all_present = prefix_present
        tail_mass = self.survival(times[-1])
        if tail_mass > 0.0:
            terms.append(math.log(tail_mass) + log_all_present)

        if not terms:
            return 0.0

        # Log-sum-exp for the normaliser; the numerator is the surviving
        # branch evaluated at t_now.
        m = max(terms)
        log_z = m + math.log(sum(math.exp(x - m) for x in terms))
        t_now = max(t_now_s, times[-1])
        log_alive = math.log(max(1e-300, self.survival(t_now))) + log_all_present
        return float(min(1.0, math.exp(log_alive - log_z)))


class Tier:
    """Three-tier taxonomy from the long-term mapping literature."""

    DYNAMIC = "dynamic"          # moves within a single session
    SEMI_STATIC = "semi_static"  # stationary per visit, moves between visits
    STATIC = "static"            # unchanged across every visit
    UNKNOWN = "unknown"          # not enough visits to decide


class SemanticKind:
    """What *sort of thing* this is.

    Kept strictly separate from persistence tier and motion state. These are
    four orthogonal axes and conflating them is how a person who sat still for
    a whole visit gets filed as building structure:

        class            person | chair | desk | laptop | backpack
        semantic_kind    person | furniture | fixture | movable_object
        persistence_tier static | semi_static | dynamic
        motion_state     moving | stationary | unknown

    Semantic kind is derived from the detector class and *never* from motion.
    A stationary human is still a person; a backpack is not furniture; a desk
    is furniture, not a fixture.
    """

    PERSON = "person"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"          # large, movable with effort
    FIXTURE = "fixture"              # built into the building
    MOVABLE = "movable_object"       # small, portable
    UNKNOWN = "unknown"


CLASS_TO_KIND: dict[str, str] = {
    "person": SemanticKind.PERSON,
    **{c: SemanticKind.ANIMAL for c in (
        "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe",
    )},
    **{c: SemanticKind.VEHICLE for c in (
        "car", "truck", "bus", "bicycle", "motorcycle", "train", "boat",
        "airplane",
    )},
    **{c: SemanticKind.FURNITURE for c in (
        "chair", "couch", "bed", "dining table", "bench", "tv",
    )},
    **{c: SemanticKind.FIXTURE for c in (
        "refrigerator", "oven", "sink", "toilet", "microwave", "toaster",
        "door", "window", "wall",
    )},
    **{c: SemanticKind.MOVABLE for c in (
        "laptop", "mouse", "keyboard", "cell phone", "book", "cup", "bottle",
        "bowl", "backpack", "handbag", "suitcase", "umbrella", "remote",
        "scissors", "vase", "clock", "potted plant", "wine glass", "fork",
        "knife", "spoon",
    )},
}


def semantic_kind(cls: str) -> str:
    return CLASS_TO_KIND.get(cls, SemanticKind.UNKNOWN)


# Semantic kind constrains which persistence tiers are even reachable. This is
# what encodes "a person is never world structure" and "a backpack is never a
# fixture" -- statements about the kind of thing, not about how often it
# happened to be observed.
ALLOWED_TIERS: dict[str, set[str]] = {
    SemanticKind.PERSON: {Tier.DYNAMIC},
    SemanticKind.ANIMAL: {Tier.DYNAMIC},
    # A car parked in the same spot daily is genuinely semi-static, but it is
    # never part of the building.
    SemanticKind.VEHICLE: {Tier.DYNAMIC, Tier.SEMI_STATIC, Tier.UNKNOWN},
    SemanticKind.MOVABLE: {Tier.DYNAMIC, Tier.SEMI_STATIC, Tier.UNKNOWN},
    SemanticKind.FURNITURE: {
        Tier.DYNAMIC, Tier.SEMI_STATIC, Tier.STATIC, Tier.UNKNOWN,
    },
    SemanticKind.FIXTURE: {
        Tier.DYNAMIC, Tier.SEMI_STATIC, Tier.STATIC, Tier.UNKNOWN,
    },
    SemanticKind.UNKNOWN: {
        Tier.DYNAMIC, Tier.SEMI_STATIC, Tier.STATIC, Tier.UNKNOWN,
    },
}

# Where to fall back when the computed tier is not permitted for this kind.
_DEMOTE = {Tier.STATIC: Tier.SEMI_STATIC, Tier.SEMI_STATIC: Tier.DYNAMIC}


def classify(
    *,
    n_visits: int,
    hit_rate: float,
    kind: str = SemanticKind.UNKNOWN,
    observed_moving: bool = False,
    p_miss: float = 0.30,
    static_ratio: float = 0.75,
    min_visits: int = 3,
) -> str:
    """Assign a persistence tier, constrained by semantic kind.

    Tier and persistence answer different questions and must not be conflated:

        tier        -- how does this object *move*?      (behaviour)
        persistence -- is it there *right now*?          (current belief)

    A laptop taken off a desk has low persistence but is still semi-static: it
    is the kind of thing that moves between visits, and it has not become a
    person. Deriving the tier from persistence would file every removed object
    under "dynamic" and hide real changes.

    So the discriminator for static vs semi-static is *consistency of presence
    across visits* (hit rate), not presence today.

    Semantic kind then acts as a hard constraint. Motion is *evidence* that
    something is dynamic, but its absence is not evidence that something is
    furniture -- a person who sat still for an entire visit would otherwise
    score a perfect hit rate and be filed as building structure. That is why
    `kind` overrides rather than merely contributes.
    """
    allowed = ALLOWED_TIERS.get(kind, ALLOWED_TIERS[SemanticKind.UNKNOWN])

    if observed_moving:
        tier = Tier.DYNAMIC
    elif n_visits < min_visits:
        # One or two visits cannot distinguish "always here" from "here now".
        tier = Tier.UNKNOWN
    elif hit_rate >= static_ratio * (1.0 - p_miss):
        # The bar for "static" is relative to what a *perfectly* permanent
        # object could actually score, which the detector caps at (1 - p_miss).
        # An absolute threshold like 0.9 is unreachable at a 30% miss rate, so
        # nothing would qualify and one unlucky miss would demote a wall.
        tier = Tier.STATIC
    else:
        tier = Tier.SEMI_STATIC

    # Demote until the tier is one this kind of thing is allowed to hold.
    while tier not in allowed:
        nxt = _DEMOTE.get(tier)
        if nxt is None or nxt == tier:
            return Tier.DYNAMIC
        tier = nxt
    return tier
