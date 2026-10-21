"""``LLMCachedClient`` — the consumer side for cases a cache may answer.

``lookup`` asks for a case by key alone; a miss comes back with a ``None``
answer and only then is ``submit`` worth its features. Both are
fire-and-forget; responses arrive in completion order and are read with
``receive``.
"""

import logging
from typing import Callable

from .answer_cache import AnswerCache
from .client import LLMBaseClient
from .config import LLMBaseConfig

logger = logging.getLogger(__name__)

Feature = tuple[str, str]


class LLMCachedClient(LLMBaseClient):
    """Answers from its own cache first, then asks the service."""

    def __init__(self, config: LLMBaseConfig, host: str | None,
                 open_cache: Callable[[], AnswerCache],
                 server_job: "int | str | None" = None) -> None:
        """Connect to the service, then open the cache read on this side.

        Opening after connecting lets the cache see whatever the server
        wrote before it opened the client port.
        """
        self._counters = {"local": 0, "forwarded": 0, "unanswered": 0}
        super().__init__(config, host, server_job)
        if not host:
            logger.info("LLMClient: no host — the cache answers alone.")
        self._cache = open_cache()

    def counters(self) -> dict:
        """Local cache hits and forwarded lookups so far."""
        return dict(self._counters)

    def lookup(self, key: tuple) -> None:
        """Ask for a case the cache may already hold.

        A hit never reaches the service. A miss is asked of the service,
        whose reply is ``None`` when it lacks the case too.
        """
        answer = self._cache.lookup(key)
        if answer is not None:
            self._ready.append((key, answer))
            self._counters["local"] += 1
            return
        if self._conn is None:  # cache-only: the miss is the final verdict
            self._ready.append((key, None))
            self._counters["unanswered"] += 1
            return
        self._conn.send(("lookup", key))
        self._counters["forwarded"] += 1

    def submit(self, key: tuple, features: list[Feature]) -> None:
        """Send one case; returns immediately. The answer comes back under ``key``."""
        if self._conn is None:
            raise RuntimeError("LLMClient: no host — nothing to submit to.")
        self._conn.send(("submit", key, features))
