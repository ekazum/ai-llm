"""``PromptConsumer`` — what generating from a ready-made prompt requires.

``ServicePromptConsumer`` satisfies it when the engine is behind a service;
a project's in-process transport can too. Callers that only submit prompts
and read text depend on this rather than on either.
"""

from typing import Iterator, Protocol


class PromptConsumer(Protocol):
    """Prompt in, generated text out, one or many at a time."""

    def generate_from_prompt(self, key: tuple, prompt: str,
                             max_tokens: "int | None" = None) -> str:
        """The text generated for one prompt, under its own reply budget."""

    def generate_batch_from_prompts(
        self, items: list[tuple[tuple, str]],
        max_tokens: "int | None" = None,
    ) -> "Iterator[list[tuple[tuple, str]]]":
        """Every prompt submitted before waiting; batches yielded as they land."""

    def answer_text(self, key: tuple) -> "str | None":
        """The retained text of a generated key, once it has returned."""
