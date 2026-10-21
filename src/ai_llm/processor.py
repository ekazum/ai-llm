"""``LLMProcessor`` — the CPU task thread that turns requests into GPU work.

Consumes ``(key, [(feature name, source text), ...])`` requests, answers
what the cache holds, and routes each miss to a GPU task as a prompt.
"""

import logging
import queue

from .answer_cache import AnswerCache

logger = logging.getLogger(__name__)

Feature = tuple[str, str]


class LLMProcessor:
    """Builds prompts and moves work between the GPU worker queues."""

    # Requests between progress lines.
    _LOG_EVERY: int = 1000

    def __init__(
        self,
        question: str,
        request_queue: queue.Queue | None = None,
    ) -> None:
        self._question = question
        # Empty when only prompt building is used.
        self._request_queue: queue.Queue = request_queue or queue.Queue()

    def build_prompt(self, features: list[Feature]) -> str:
        """Return the prompt for one case, sections in the order given."""
        sections = [f"{name}:\n{text}" for name, text in features]
        return "\n\n".join(sections + [self._question])

    def serve(
        self,
        gpu_queue: queue.Queue,
        response_queue: queue.Queue,
        cache: AnswerCache | None,
        stats=None,
    ) -> None:
        """Serving loop: one request at a time, cache hit or GPU task.

        A hit goes straight to the sender. A lookup that misses is answered
        with ``None``, and the asker sends the prompt it now knows is needed.
        """
        n_requests = n_hits = 0
        while True:
            item = self._request_queue.get()
            if item is None:
                break
            key, features, prompt, budget = item
            answer = None if cache is None else cache.lookup(key)
            if answer is not None:
                response_queue.put((key, answer))
                n_hits += 1
            elif prompt is not None:
                # Built by the asker, under the budget it asked for.
                gpu_queue.put((key, prompt, budget))
            elif features is None:
                # A lookup that missed: the asker sends the prompt next.
                response_queue.put((key, None))
            else:
                gpu_queue.put((key, self.build_prompt(features)))
            n_requests += 1
            if stats is not None and features is None:
                # One consultation is one lookup; its retry is counted below.
                stats.event(requests=1,
                            cache_hits=int(answer is not None),
                            cache_misses=int(answer is None))
            elif stats is not None:
                stats.event(dispatched=1)
            if n_requests % self._LOG_EVERY == 0:
                logger.info(
                    "LLMProcessor: %d requests routed, %d cache hits, "
                    "%d still queued.",
                    n_requests, n_hits, self._request_queue.qsize(),
                )
        logger.info("LLMProcessor: serving done.")
