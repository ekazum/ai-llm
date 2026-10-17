"""``LLMBaseClient`` — the consumer side of the service: the connection.

Connects to the serving CPU task at the given host and the configured port.
``submit_prompt`` is fire-and-forget; responses arrive on the same
connection in completion order and are read with ``receive``.
"""

import logging
import time
from collections import deque
from multiprocessing.connection import Client, Connection

from ai_llm.config import LLMBaseConfig

logger = logging.getLogger(__name__)

# The client port opens only once the server is ready.
_CONNECT_RETRY_SECONDS = 3600.0
_CONNECT_POLL_SECONDS = 5.0


class LLMBaseClient:
    """Submits prompts and receives their answers."""

    def __init__(self, config: LLMBaseConfig, host: str | None,
                 server_job: "int | str | None" = None) -> None:
        """Connect to the service; no host leaves the client unconnected.

        ``server_job`` derives the server's run-scoped port; None means
        the server runs in this very job.
        """
        # Answers produced on this side, waiting to be received.
        self._ready: deque = deque()
        self._conn: Connection | None = None
        if host:
            port = config.client_port(server_job)
            deadline = time.monotonic() + _CONNECT_RETRY_SECONDS
            while True:
                try:
                    self._conn = Client((host, port))
                    break
                except (ConnectionRefusedError, OSError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(_CONNECT_POLL_SECONDS)
            self._conn.send(("hello", "consumer"))
            logger.info("LLMClient: connected to %s:%d.", host, port)

    def close(self) -> None:
        """Drop the connection to the service. Safe to call twice."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def submit_prompt(self, key: tuple, prompt: str,
                      max_tokens: int) -> None:
        """Send one ready-made prompt under its own reply budget.

        The answer comes back as text under ``key``.
        """
        if self._conn is None:
            raise RuntimeError("LLMClient: no host — nothing to submit to.")
        self._conn.send(("prompt", key, prompt, int(max_tokens)))

    def receive(self) -> tuple:
        """Return the next ``(key, answer)``; the answer is ``None`` on a miss."""
        if self._ready:
            return self._ready.popleft()
        if self._conn is None:
            raise RuntimeError("LLMClient: no host and nothing ready.")
        return self._conn.recv()

    def poll(self, timeout: float = 0.0) -> bool:
        """Whether a finished response is ready to ``receive``."""
        if self._ready:
            return True
        return False if self._conn is None else self._conn.poll(timeout)
