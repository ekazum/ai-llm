"""``ServicePromptConsumer`` — ready-made prompts over the service.

The transport when the engine is behind a service: it submits prompts the
caller built and returns the generated text. Its own connection, so the
answers it waits on never cross another consumer's bookkeeping, and the
``PromptConsumer`` methods, so an in-process consumer can stand in for it.
"""

import logging
import time
from typing import Iterator

from ai_llm.client import LLMBaseClient
from ai_llm.config import LLMBaseConfig

logger = logging.getLogger(__name__)

# How long a batch may go without a single answer before it is called lost.
# Generation is minutes at worst; silence past this is the service gone, not
# a slow reply, and waiting it out costs the job its whole wall clock.
GENERATE_TIMEOUT = 900.0
# How often the wait wakes to check, so the deadline is observed promptly.
POLL_SECONDS = 5.0


class ServicePromptConsumer:
    """Prompt in, text out, over one connection to the service."""

    def __init__(self, config: LLMBaseConfig, host: str,
                 server_job: "int | str | None" = None) -> None:
        """Dial the service this consumer generates through."""
        self._client = LLMBaseClient(config, host, server_job)
        self._max_tokens = config.LLM_MAX_TOKENS
        self._texts: dict[tuple, str] = {}

    def generate_from_prompt(self, key: tuple, prompt: str,
                             max_tokens: "int | None" = None) -> str:
        """One prompt; blocks until its own answer returns."""
        for batch in self.generate_batch_from_prompts([(key, prompt)],
                                                      max_tokens):
            for finished, answer in batch:
                if finished == key:
                    return answer
        raise RuntimeError(f"no answer returned for {key}")

    def generate_batch_from_prompts(
        self, items: list[tuple[tuple, str]],
        max_tokens: "int | None" = None,
    ) -> "Iterator[list[tuple[tuple, str]]]":
        """Every prompt submitted before waiting; answers yielded as they land.

        The service returns them one message at a time, so each yield holds
        what one receive brought back.
        """
        budget = max_tokens or self._max_tokens
        pending = {key for key, _ in items}
        for key, prompt in items:
            self._client.submit_prompt(key, prompt, budget)
        while pending:
            self._await_answer(len(pending))
            finished, answer = self._client.receive()
            self._texts[finished] = answer
            if finished in pending:
                pending.discard(finished)
                yield [(finished, answer[0])]

    def _await_answer(self, outstanding: int) -> None:
        """Wait for one answer to arrive; raise if none ever does.

        Without this the phase blocks on a socket that a stopped service
        never answers, and the job spends its wall clock on it.
        """
        deadline = time.monotonic() + GENERATE_TIMEOUT
        while not self._client.poll(POLL_SECONDS):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"generation timed out after {GENERATE_TIMEOUT:.0f}s "
                    f"with {outstanding} prompts outstanding"
                )

    def answer_text(self, key: tuple) -> "str | None":
        """The retained answer text of a generated key."""
        answer = self._texts.get(key)
        return None if answer is None else answer[0]

    def close(self) -> None:
        """Give the connection back."""
        self._client.close()
