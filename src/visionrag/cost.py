"""Compute accounting.

This is not instrumentation bolted on for a results table -- CPU cost is the
x-axis of the paper's central claim, so it is measured from the first commit
and every stage is attributed separately.

Two clocks are recorded because they answer different questions:

* ``process_time`` sums CPU time across all threads of the process. This is the
  honest "how much compute did this consume" number and is what the
  accuracy-vs-compute curve uses. It is *not* comparable to wall time on a
  16-core box -- a well-threaded detector can burn 8 CPU-seconds in 1 wall
  second.
* ``perf_counter`` is wall time, which is what latency claims (p50/p95) need.

Reporting only one of these is the most common way a systems paper gets
picked apart.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class StageStats:
    calls: int = 0
    cpu_s: float = 0.0
    wall_s: float = 0.0
    wall_samples: list[float] = field(default_factory=list)

    def percentile(self, p: float) -> float:
        if not self.wall_samples:
            return 0.0
        ordered = sorted(self.wall_samples)
        # Nearest-rank; exact enough at our sample counts and avoids
        # interpolation arguments in review.
        k = max(0, min(len(ordered) - 1, int(round(p / 100.0 * len(ordered))) - 1))
        return ordered[k]


class CostLedger:
    """Accumulates per-stage compute across a run."""

    def __init__(self, keep_samples: bool = True) -> None:
        self._stages: dict[str, StageStats] = defaultdict(StageStats)
        self._keep_samples = keep_samples
        self._t0_cpu = time.process_time()
        self._t0_wall = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        c0, w0 = time.process_time(), time.perf_counter()
        try:
            yield
        finally:
            dc = time.process_time() - c0
            dw = time.perf_counter() - w0
            s = self._stages[name]
            s.calls += 1
            s.cpu_s += dc
            s.wall_s += dw
            if self._keep_samples:
                s.wall_samples.append(dw)

    @property
    def total_cpu_s(self) -> float:
        return time.process_time() - self._t0_cpu

    @property
    def total_wall_s(self) -> float:
        return time.perf_counter() - self._t0_wall

    def stages(self) -> dict[str, StageStats]:
        return dict(self._stages)

    def summary(self, video_duration_s: float | None = None) -> dict:
        """Machine-readable summary. `video_duration_s` enables the headline
        normalised metric: CPU-seconds consumed per minute of video ingested."""
        out: dict = {
            "total_cpu_s": round(self.total_cpu_s, 4),
            "total_wall_s": round(self.total_wall_s, 4),
            "stages": {},
        }
        for name, s in sorted(self._stages.items()):
            out["stages"][name] = {
                "calls": s.calls,
                "cpu_s": round(s.cpu_s, 4),
                "wall_s": round(s.wall_s, 4),
                "wall_ms_mean": round(1000 * s.wall_s / s.calls, 3) if s.calls else 0.0,
                "wall_ms_p50": round(1000 * s.percentile(50), 3),
                "wall_ms_p95": round(1000 * s.percentile(95), 3),
            }
        if video_duration_s and video_duration_s > 0:
            out["cpu_s_per_video_min"] = round(
                self.total_cpu_s / (video_duration_s / 60.0), 4
            )
            out["realtime_factor"] = round(video_duration_s / self.total_wall_s, 3)
        return out


def hardware_fingerprint() -> dict:
    """Recorded with every run. A CPU-efficiency claim on unnamed hardware is
    not a claim, so this travels with the results."""
    import os
    import platform

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
    }
    try:  # physical core count is more meaningful but needs an optional dep
        import psutil  # type: ignore

        info["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:
        info["physical_cores"] = None
    return info


def latency_report(samples_s: list[float]) -> dict:
    """End-to-end latency distribution (PRD 8.3)."""
    if not samples_s:
        return {}
    ms = sorted(x * 1000 for x in samples_s)

    def pct(p: float) -> float:
        k = max(0, min(len(ms) - 1, int(round(p / 100.0 * len(ms))) - 1))
        return round(ms[k], 3)

    return {
        "n": len(ms),
        "mean_ms": round(statistics.fmean(ms), 3),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": round(ms[-1], 3),
    }
