"""``AnswerCache`` — what keeping generated answers requires.

The key is any tuple the project chooses; ai-llm passes it through and never
reads it. The answer is whatever the project's engine returns for a key; its
type is the protocol's parameter, so ``AnswerCache[tuple[str, float]]`` looks
up a ``tuple[str, float]``. A bare ``AnswerCache`` keeps any answer.
"""

from typing import Protocol, TypeVar

AnswerT = TypeVar("AnswerT")
# The (text, *floats) answers of the built-in stores.
TupleAnswerT = TypeVar("TupleAnswerT", bound=tuple)


class AnswerCache(Protocol[AnswerT]):
    """Answers kept by key: looked up before generating, updated after."""

    def lookup(self, key: tuple) -> "AnswerT | None":
        """The answer kept for a key, or ``None``."""

    def update(self, key: tuple, answer: AnswerT) -> bool:
        """Keep an answer; ``False`` when this cache does not keep the key."""

    def flush(self) -> None:
        """Persist what ``update`` has kept so far."""
