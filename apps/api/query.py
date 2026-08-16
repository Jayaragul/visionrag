"""Evidence-grounded question answering.

Deliberately *not* LLM-backed at this stage. Retrieval, grounding and
abstention are the parts that carry the claim; prose generation is a
presentation layer that can be added behind an adapter later. Keeping it out
for now also means no session data leaves the machine.

Every answer is assembled from rows that were actually retrieved, so the
citations cannot drift from the evidence. When nothing is retrieved above the
confidence floor, the system says so instead of guessing -- abstention is a
feature here, and it is what E4 measures.
"""

from __future__ import annotations

import re

from visionrag.memory.store import MemoryStore
from visionrag.types import EventType

# Question phrasing -> event types worth retrieving.
_INTENT_PATTERNS: list[tuple[str, list[EventType]]] = [
    (r"\b(appear|arrive|come|enter|show up)\w*\b",
     [EventType.APPEARED, EventType.REAPPEARED]),
    (r"\b(disappear|leave|left|gone|vanish|exit)\w*\b",
     [EventType.DISAPPEARED]),
    (r"\b(approach|toward|closer|near)\w*\b",
     [EventType.APPROACHED, EventType.NEAR, EventType.REMAINED_NEAR]),
    (r"\b(away|apart|separat|retreat)\w*\b", [EventType.MOVED_AWAY]),
    (r"\b(stop|stat|still|halt|pause)\w*\b", [EventType.STOPPED]),
    (r"\b(mov|walk|start|go)\w*\b",
     [EventType.STARTED_MOVING, EventType.STOPPED]),
    (r"\b(camera|shake|handheld)\w*\b",
     [EventType.CAMERA_MOVED, EventType.CAMERA_STABILISED]),
]

_PHRASING = {
    EventType.APPEARED: "{who} appeared",
    EventType.REAPPEARED: "{who} reappeared",
    EventType.DISAPPEARED: "{who} went out of view",
    EventType.APPROACHED: "{who} moved closer together",
    EventType.MOVED_AWAY: "{who} moved apart",
    EventType.NEAR: "{who} came near each other",
    EventType.REMAINED_NEAR: "{who} stayed near each other",
    EventType.STOPPED: "{who} stopped",
    EventType.STARTED_MOVING: "{who} started moving",
    EventType.CAMERA_MOVED: "the camera started moving",
    EventType.CAMERA_STABILISED: "the camera became steady",
}


def _intents(question: str) -> list[str]:
    q = question.lower()
    types: list[str] = []
    for pattern, event_types in _INTENT_PATTERNS:
        if re.search(pattern, q):
            types.extend(t.value for t in event_types)
    return list(dict.fromkeys(types))  # de-duplicate, keep order


def _time_filter(question: str, elapsed_ms: int) -> tuple[int | None, int | None]:
    """Resolve coarse relative time references. Anything unrecognised returns
    an unbounded range rather than a guessed one."""
    q = question.lower()
    m = re.search(r"last (\d+)\s*(second|sec|minute|min)", q)
    if m:
        n = int(m.group(1))
        span = n * (60_000 if m.group(2).startswith("min") else 1000)
        return max(0, elapsed_ms - span), None
    if re.search(r"\b(just now|recently|last|latest)\b", q):
        return max(0, elapsed_ms - 30_000), None
    if re.search(r"\b(begin|start|first|initially)\b", q):
        return None, 30_000
    return None, None


def _describe(who: list[int], event_type: str) -> str:
    try:
        kind = EventType(event_type)
    except ValueError:
        return f"{event_type} involving {who}"
    if not who:
        subject = "something"
    elif len(who) == 1:
        subject = f"object #{who[0]}"
    else:
        subject = " and ".join(f"object #{w}" for w in who)
    return _PHRASING.get(kind, f"{{who}} {event_type}").format(who=subject)


def answer_question(
    store: MemoryStore,
    question: str,
    run_id: int,
    elapsed_ms: int,
    min_confidence: float = 0.3,
    max_events: int = 8,
) -> dict:
    """Retrieve, then compose an answer strictly from what was retrieved."""
    types = _intents(question)
    t_start, t_end = _time_filter(question, elapsed_ms)

    events = store.timeline(
        run_id=run_id,
        t_start_ms=t_start,
        t_end_ms=t_end,
        types=types or None,
        min_confidence=min_confidence,
        limit=200,
    )

    # Fall back to an unfiltered window before abstaining: a question whose
    # wording matched no intent pattern is a failure of the pattern list, not
    # evidence that nothing happened.
    if not events and types:
        events = store.timeline(
            run_id=run_id,
            t_start_ms=t_start,
            t_end_ms=t_end,
            min_confidence=min_confidence,
            limit=200,
        )

    if not events:
        return {
            "answer": "I don't have evidence for that in this session.",
            "confidence": 0.0,
            "abstained": True,
            "evidence_ids": [],
            "time_range": None,
            "events": [],
            "limitations": "No stored events matched the question above the "
                           "confidence threshold.",
        }

    events.sort(key=lambda e: e["confidence"], reverse=True)
    top = events[:max_events]
    top.sort(key=lambda e: e["t_start_ms"])

    parts = [
        f"{_describe(e['participants'], e['type'])} at "
        f"{e['t_start_ms'] / 1000:.1f}s"
        for e in top
    ]
    answer = "; ".join(parts).capitalize() + "."

    evidence_ids: list[str] = []
    for e in top:
        evidence_ids.append(f"event_{e['id']}")
        evidence_ids.extend(f"frame_{f}" for f in e["evidence"][:2])

    # Answer confidence is the weakest link among cited events, not the mean:
    # a claim built from several events is only as good as its shakiest one.
    confidence = min(e["confidence"] for e in top)

    limitations = []
    if any(e["ego_suspect"] for e in top):
        limitations.append(
            "Some cited events could be explained by camera motion."
        )
    if len(events) > len(top):
        limitations.append(f"{len(events) - len(top)} lower-ranked events omitted.")
    if not types:
        limitations.append("Question intent was unclear; returned a general summary.")

    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "abstained": False,
        "evidence_ids": evidence_ids,
        "time_range": [
            round(min(e["t_start_ms"] for e in top) / 1000, 2),
            round(max(e["t_end_ms"] for e in top) / 1000, 2),
        ],
        "events": top,
        "limitations": " ".join(limitations) or None,
    }
