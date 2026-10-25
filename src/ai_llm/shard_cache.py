"""``ShardCache`` — the answers this run generated, flushed as npy shards.

An answer is ``(text, *floats)`` in ``CacheSchema`` order, kept by a key of
the schema's width. Each flush writes one keys, texts and floats shard that
a ``CacheBuilder`` later folds into the store.
"""

import logging
import time
from pathlib import Path

import numpy as np

from .answer_cache import AnswerCache
from .cache_schema import CacheSchema

logger = logging.getLogger(__name__)


class ShardCache(AnswerCache):
    """Accumulates this run's answers."""

    def __init__(self, schema: CacheSchema, directory: str | Path,
                 source: str = "local", flush_every: int = 4096,
                 flush_seconds: float = 30.0) -> None:
        """``source`` names the shards; a flush runs at ``flush_every``
        answers held or ``flush_seconds`` since the last."""
        self._schema = schema
        self._dir = Path(directory)
        self._source = source
        self._flush_every = flush_every
        self._flush_seconds = flush_seconds
        self._width = len(schema.keys)
        self._live: dict[tuple, tuple] = {}
        self._pending: list[tuple[tuple, tuple]] = []
        self._flushed = time.monotonic()
        self._shard = 0
        self._counters = {"hits": 0, "misses": 0, "writes": 0}

    def lookup(self, key: tuple) -> tuple | None:
        """``AnswerCache.lookup``: a key of another width is never held."""
        if len(key) != self._width:
            return None
        answer = self._live.get(key)
        self._counters["hits" if answer is not None else "misses"] += 1
        return answer

    def update(self, key: tuple, answer: tuple) -> bool:
        """``AnswerCache.update``: keep an answer and queue it for the build."""
        if len(key) != self._width:
            return False
        text, *floats = answer
        answer = (str(text), *(float(value) for value in floats))
        self._live[key] = answer
        self._pending.append((key, answer))
        self._counters["writes"] += 1
        if (len(self._pending) >= self._flush_every
                or time.monotonic() - self._flushed >= self._flush_seconds):
            self.flush()
        return True

    def flush(self) -> None:
        """Write the queued answers as one shard the next build picks up."""
        self._flushed = time.monotonic()
        if not self._pending:
            return
        keys = np.array([key for key, _ in self._pending], dtype=np.int64)
        texts = np.array([answer[0] for _, answer in self._pending])
        floats = np.array([answer[1:] for _, answer in self._pending],
                          dtype=np.float32)
        name = f"{self._source}_{self._shard}"
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._dir / f"{self._schema.key_shard}shard{name}.npy", keys)
        np.save(self._dir / f"{self._schema.text_shard}shard{name}.npy", texts)
        np.save(self._dir / f"{self._schema.float_shard}shard{name}.npy",
                floats)
        logger.info("ShardCache: %d answers flushed to shard %s.",
                    len(keys), name)
        self._pending.clear()
        self._shard += 1

    def counters(self) -> dict:
        """Hits, misses and writes so far, and the answers held."""
        return dict(self._counters, live=len(self._live))
