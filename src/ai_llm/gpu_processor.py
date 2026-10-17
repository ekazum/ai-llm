"""``LLMGPUProcessor`` — moves work through the GPU process's workers.

Several run per GPU process. Each tokenizes on the CPU, submits without
waiting, and drains its own result queue every pass, forwarding finished
answers to the sender. ``LLM_GPU_MAX_IN_FLIGHT`` bounds the items one
processor holds.
"""

import logging
import queue
import threading
import time
from typing import Any, cast

from transformers import AutoTokenizer

from ai_runtime.batching import BatchRequest as LLMGPURequest

from ai_llm.config import LLMBaseConfig

logger = logging.getLogger(__name__)


class LLMGPUProcessor:
    """Keeps the GPU workers fed and assembles what they return."""

    # How long ``run`` blocks on the queue between shutdown checks.
    _QUEUE_POLL_TIMEOUT: float = 0.1

    def __init__(
        self,
        config: LLMBaseConfig,
        in_queue: queue.Queue,
        gen_queue: queue.Queue,
        out_queue: queue.Queue,
        shutdown_event: threading.Event,
    ) -> None:
        self._config = config
        self._in_queue = in_queue
        self._gen_queue = gen_queue
        self._out_queue = out_queue
        self._shutdown_event = shutdown_event
        # This thread's own result queue, shared by all its in-flight items.
        self._gen_result: queue.Queue = queue.Queue()
        self._in_flight = 0

    def run(self) -> None:
        """Forward finished work and submit new work until shutdown."""
        # The auto class's stub has no return type; it builds a tokenizer.
        tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(self._config.LLM_MODEL)
        while not self._shutdown_event.is_set():
            self._emit_finished()
            if self._in_flight >= self._config.LLM_GPU_MAX_IN_FLIGHT:
                time.sleep(self._QUEUE_POLL_TIMEOUT)
                continue
            keys, prompts, budgets = self._take_work()
            if keys:
                self._submit(tokenizer, keys, prompts, budgets)
        logger.info("LLMGPUProcessor: done.")

    def _take_work(self) -> tuple[list, list, list]:
        """Take prompts up to the in-flight room, waiting for the first."""
        keys: list = []
        prompts: list = []
        budgets: list = []
        room = self._config.LLM_GPU_MAX_IN_FLIGHT - self._in_flight
        try:
            key, prompt, budget = self._unpack(
                self._in_queue.get(timeout=self._QUEUE_POLL_TIMEOUT))
        except queue.Empty:
            return keys, prompts, budgets
        while True:
            keys.append(key)
            prompts.append(prompt)
            budgets.append(budget)
            if len(keys) >= room:
                return keys, prompts, budgets
            try:
                key, prompt, budget = self._unpack(self._in_queue.get_nowait())
            except queue.Empty:
                return keys, prompts, budgets

    def _unpack(self, item: tuple) -> tuple:
        """An item may name its own reply budget; else the config's applies."""
        if len(item) == 3:
            return item
        key, prompt = item
        return key, prompt, self._config.LLM_MAX_TOKENS

    def _submit(self, tokenizer, keys: list, prompts: list,
                budgets: list) -> None:
        """Tokenize prompts and hand them to the generation worker.

        Each prompt keeps the window its own reply leaves it, so a long
        reply never narrows another item's room. Over-length prompts keep
        their last tokens.
        """
        n_truncated = 0
        smallest_limit = self._config.LLM_MAX_MODEL_LEN
        items = []
        for prompt, budget in zip(prompts, budgets):
            limit = self._config.LLM_MAX_MODEL_LEN - budget - 64
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                # Plain id list rather than the default BatchEncoding.
                return_dict=False,
            )
            if len(ids) > limit:
                ids = ids[-limit:]
                n_truncated += 1
                smallest_limit = min(smallest_limit, limit)
            items.append(ids)
        if n_truncated:
            logger.warning(
                "%d of %d prompts truncated to %d tokens.",
                n_truncated, len(prompts), smallest_limit,
            )
        self._gen_queue.put(
            LLMGPURequest(keys=keys, items=items, sink=self._gen_result,
                          budgets=budgets)
        )
        self._in_flight += len(keys)

    def _emit_finished(self) -> None:
        """Hand finished answers back for sending."""
        keys, answers = self._drain(self._gen_result)
        if not keys:
            return
        self._out_queue.put((keys, answers))
        self._in_flight -= len(keys)

    @staticmethod
    def _drain(result_queue: queue.Queue) -> tuple[list, list]:
        """Take everything queued now, as keys and results."""
        keys: list = []
        results: list = []
        while True:
            try:
                key, result = result_queue.get_nowait()
            except queue.Empty:
                return keys, results
            keys.append(key)
            results.append(result)
