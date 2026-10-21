"""``AnswerCache`` — what keeping generated answers requires.

The key is any tuple the project chooses; ai-llm passes it through and never
reads it. The answer is whatever the project's engine returns for a key.
"""

from typing import Any, Protocol


class AnswerCache(Protocol):
    """Answers kept by key: looked up before generating, updated after."""

    def lookup(self, key: tuple) -> "Any | None":
        """The answer kept for a key, or ``None``."""

    def update(self, key: tuple, answer: Any) -> bool:
        """Keep an answer; ``False`` when this cache does not keep the key."""

    def flush(self) -> None:
        """Persist what ``update`` has kept so far."""
