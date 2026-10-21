"""``LLMCollector`` — the service thread that finishes completed work.

Receives ``(keys, answers)`` batches from the GPU tasks, keeps the answers
in the cache, and hands the responses to the server thread for sending.
"""

import logging
import queue

from .answer_cache import AnswerCache

logger = logging.getLogger(__name__)


class LLMCollector:
    """Finishes completed work: cache the answer, hand to send."""

    def __init__(
        self,
        cache: AnswerCache | None,
        done_queue: queue.Queue,
        response_queue: queue.Queue,
        stats=None,
    ) -> None:
        """Store the cache the answers are kept in."""
        self._cache = cache
        self._done_queue = done_queue
        self._response_queue = response_queue
        self._stats = stats

    def run(self) -> None:
        """Consume finished ``(keys, answers)`` batches until the ``None``
        sentinel."""
        while True:
            item = self._done_queue.get()
            if item is None:
                break
            keys, answers = item
            cached = 0
            for key, answer in zip(keys, answers):
                # The cache decides which keys are worth keeping.
                if self._cache is not None and self._cache.update(key, answer):
                    cached += 1
                self._response_queue.put((key, answer))
            if self._stats is not None:
                self._stats.event(answers_finished=len(keys),
                                  cache_writes=cached)
            logger.info("LLMCollector: %d responses finished.", len(keys))
        if self._cache is not None:
            self._cache.flush()
        logger.info("LLMCollector: done.")
