"""Machine terms to human sentences.

The system's internal vocabulary is precise and unreadable: `remained_near
tracks=[1,2] conf=0.80`, `semi_static persistence=0.019`. That is the right
vocabulary for the event rules and the wrong one for a person holding a phone.

One module so the phone, the dashboard and the query answers all say the same
thing the same way. Divergent phrasing across surfaces is how a product starts
feeling like three products.

Two rules throughout:

* **Never overstate.** Belief is rendered as hedged language ("probably still
  here") rather than a number pretending to be a fact. The uncertainty is
  real; hiding it would be a lie, and showing it as `0.62` is not an answer.
* **Never invent.** Nothing here adds information that was not in the record.
  Phrasing translates; it does not embellish.
"""

from __future__ import annotations

# Articles for readable sentences. Only classes we phrase often need an entry;
# anything missing falls back to a bare noun, which reads acceptably.
_A = {
    "person": "someone", "cat": "a cat", "dog": "a dog",
    "car": "a car", "bicycle": "a bicycle", "truck": "a truck",
    "chair": "a chair", "laptop": "a laptop", "cup": "a cup",
    "bottle": "a bottle", "backpack": "a backpack", "handbag": "a bag",
    "book": "a book", "cell phone": "a phone", "tv": "the screen",
    "dining table": "the table", "couch": "the couch", "bed": "the bed",
    "refrigerator": "the fridge", "potted plant": "a plant",
    "keyboard": "a keyboard", "mouse": "a mouse", "suitcase": "a suitcase",
}


def noun(cls: str, definite: bool = False) -> str:
    """`laptop` -> `a laptop`, or `the laptop` when definite."""
    if definite:
        base = cls.replace("_", " ")
        return base if base.startswith("the ") else f"the {base}"
    return _A.get(cls, cls.replace("_", " "))


def _subject(participants: list[int], classes: dict[int, str]) -> str:
    names = [noun(classes[p]) for p in participants if p in classes]
    if not names:
        return "something"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


# Event type -> sentence template. `{s}` is the subject.
_EVENTS = {
    "appeared": "{s} came into view",
    "reappeared": "{s} came back",
    "disappeared": "{s} went out of view",
    "started_moving": "{s} started moving",
    "stopped": "{s} stopped",
    "direction_changed": "{s} changed direction",
    "entered_region": "{s} entered the area",
    "left_region": "{s} left the area",
    "near": "{s} came close together",
    "approached": "{s} moved closer together",
    "moved_away": "{s} moved apart",
    "remained_near": "{s} stayed close together",
    "camera_moved": "the camera started moving",
    "camera_stabilised": "the camera settled",
}


def describe_event(event: dict, classes: dict[int, str]) -> str:
    """One readable line for a timeline entry."""
    kind = event.get("type", "")
    template = _EVENTS.get(kind)
    if template is None:
        return kind.replace("_", " ")
    subject = _subject(event.get("participants") or [], classes)
    sentence = template.format(s=subject)
    return sentence[0].upper() + sentence[1:]


def event_hint(event: dict) -> str | None:
    """Short qualifier shown beside a timeline entry, or None.

    Only surfaces things a person would act on. Confidence numbers are not
    included: a reader cannot calibrate 0.53 against 0.61, so showing both
    just adds noise.
    """
    if event.get("ego_suspect"):
        return "may be camera movement"
    attrs = event.get("attrs") or {}
    if "duration_s" in attrs:
        seconds = attrs["duration_s"]
        if seconds >= 60:
            return f"for {round(seconds / 60)} min"
        return f"for {round(seconds)}s"
    return None


# -- belief ------------------------------------------------------------
def belief(persistence: float) -> str:
    """Hedged language for a probability.

    Bands are wide on purpose. The underlying estimate is not precise enough
    to justify fine gradations, and offering five near-identical phrasings
    would imply a precision the filter does not have.
    """
    if persistence >= 0.85:
        return "still here"
    if persistence >= 0.55:
        return "probably still here"
    if persistence >= 0.25:
        return "might be gone"
    return "gone"


def belief_tone(persistence: float) -> str:
    """UI colour class, kept next to the wording so they cannot drift apart."""
    if persistence >= 0.85:
        return "ok"
    if persistence >= 0.55:
        return "mild"
    if persistence >= 0.25:
        return "warn"
    return "bad"


_KINDS = {
    "fixture": "part of the room",
    "furniture": "furniture",
    "movable_object": "moves around",
    "person": "a person",
    "animal": "an animal",
    "vehicle": "a vehicle",
    "unknown": "unrecognised",
}


def describe_kind(kind: str) -> str:
    return _KINDS.get(kind, kind.replace("_", " "))


def describe_place(label: str | None, place_id: int, n_visits: int) -> dict:
    """Title and subtitle for the place header."""
    title = label or f"Unnamed place {place_id}"
    if n_visits <= 1:
        subtitle = "First time here"
    elif n_visits == 2:
        subtitle = "Second visit"
    else:
        subtitle = f"{n_visits} visits"
    return {"title": title, "subtitle": subtitle, "named": bool(label)}


def describe_change(entry: dict, direction: str) -> str:
    """One line for the change list."""
    name = noun(entry["class"], definite=(direction == "removed"))
    if direction == "removed":
        return f"{name} is gone".capitalize()
    if direction == "added":
        return f"{name} is new here".capitalize()
    return name.capitalize()


def summarise_changes(changes: dict) -> str:
    """Headline for the change card.

    Says "nothing changed" only when there was enough coverage to know. A scan
    that saw a third of the room has not established that the rest is
    unchanged, and claiming otherwise is the failure the coverage model exists
    to prevent.
    """
    removed = len(changes.get("removed", []))
    added = len(changes.get("added", []))
    if removed == 0 and added == 0:
        return "Nothing has changed here"
    parts = []
    if removed:
        parts.append(f"{removed} thing{'s' if removed > 1 else ''} gone")
    if added:
        parts.append(f"{added} new")
    return " · ".join(parts)


def quality_note(quality: dict | None) -> str | None:
    """Plain-language reason the camera view is not good enough.

    Phrased as something the user can act on -- "too dark to see" invites
    turning on a light; "brightness 0.08 below DARK_LIMIT" does not.
    """
    if not quality or quality.get("usable"):
        return None
    fixes = {
        "too dark": "Too dark to see clearly",
        "overexposed": "Too bright — try pointing away from the light",
        "no contrast (lens covered?)": "Camera may be covered",
        "blurred": "Hold steady — the view is blurred",
        "heavily clipped": "Harsh lighting is washing out the view",
    }
    for reason in quality.get("reasons", []):
        if reason in fixes:
            return fixes[reason]
    return "Camera view is unclear"
