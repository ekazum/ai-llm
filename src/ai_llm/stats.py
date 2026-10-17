"""``LLMStats`` — the serving run's counters, owned by one thread.

The service threads only ``event(...)`` their counts onto a queue; this
thread drains it, owns the totals exclusively (no locks on the hot path),
and records them every ``write_every`` events and once more at shutdown.
The service has no natural end, so the store always reflects the run so
far. Rows carry the serving job's id, so several runs accumulate in one
store rather than a file each.
"""

import logging
import os
import queue
import time
from datetime import datetime
from pathlib import Path

from ai_llm.stats_store import LLMStatsStore

logger = logging.getLogger(__name__)


class LLMStats:
    """Drains counter events, writes the serving run's stats files."""

    SERIES_COLUMNS = ("requests", "dispatched", "answers_finished",
                      "cache_hits", "cache_misses", "cache_writes")
    SERIES_INTERVAL_SECONDS = 30.0
    HEARTBEAT_SECONDS = 60.0

    def __init__(self, path: str, write_every: int = 50) -> None:
        self._job_id = os.environ.get("SLURM_JOB_ID", "local")
        self._label = os.environ.get("LLM_RUN_LABEL", "")
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._store = LLMStatsStore(Path(path).parent)
        self._store.start_run(self._job_id, self._label, self._started_at)
        self._last_series = 0.0
        self._write_every = write_every
        self._queue: queue.Queue = queue.Queue()
        self._counters: dict[str, int] = {}
        self._gpu: dict[str, dict[str, int]] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def event(self, **counters: int) -> None:
        """Producer side: queue named counter increments."""
        self._queue.put(counters)

    def gpu_event(self, source: str, counters: dict) -> None:
        """Producer side: queue one GPU task's counters, kept per task."""
        self._queue.put({"_gpu": (source, counters)})

    def stop(self) -> None:
        """Producer side: end the stats thread."""
        self._queue.put(None)

    def run(self) -> None:
        """Drain events until the sentinel, writing as totals accumulate."""
        self.write()   # run visible (zero counters) before any traffic
        drained = 0
        while True:
            try:
                item = self._queue.get(timeout=self.HEARTBEAT_SECONDS)
            except queue.Empty:
                self.write()   # heartbeat: liveness via file mtime
                continue
            if item is None:
                break
            for name, value in item.items():
                if name == "_gpu":
                    self._add_gpu(*value)
                    continue
                self._counters[name] = self._counters.get(name, 0) + int(value)
            drained += 1
            # Current between bursts; capped during one.
            if self._queue.empty() or drained % self._write_every == 0:
                self.write()
        self.write(stopped=True)
        self._append_series(force=True)   # final point regardless of throttle
        logger.info("LLMStats: %d events, recorded in %s.",
                    drained, self._store.path)

    def _add_gpu(self, source: str, counters: dict) -> None:
        """Fold one GPU task's counters into its own totals."""
        totals = self._gpu.setdefault(str(source), {})
        for name, value in counters.items():
            totals[name] = totals.get(name, 0) + int(value)

    def write(self, stopped: bool = False) -> None:
        """Record the run's totals; append the throttled series point."""
        self._store.set_counters(self._job_id, self._counters)
        self._store.set_gpu_totals(self._job_id, self._gpu)
        if stopped:
            self._store.stop_run(
                self._job_id, datetime.now().isoformat(timespec="seconds"))
        self._append_series()

    def _append_series(self, force: bool = False) -> None:
        """Append one time-series point, at most every SERIES_INTERVAL_SECONDS."""
        now = time.time()
        if not force and now - self._last_series < self.SERIES_INTERVAL_SECONDS:
            return
        self._last_series = now
        self._store.add_series(
            self._job_id, datetime.now().isoformat(timespec="seconds"),
            {**{c: int(self._counters.get(c, 0)) for c in self.SERIES_COLUMNS},
             "gpu_answers": sum(t.get("answers", 0)
                                for t in self._gpu.values())},
        )
