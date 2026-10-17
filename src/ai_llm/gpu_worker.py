"""Answer generation, one engine spanning the GPU process's GPUs.

Runs as a thread of the GPU process; ai-runtime's ``GpuBatchStage`` does the
windowing and batching, with this worker as its ``BatchModel``. The engine
runs tensor-parallel over every GPU the job owns (vLLM spawns and manages the
NCCL ranks), sized by ``LLM_GEN_GPU_MEM_UTIL``. Items arrive as token ids.
"""

import logging

from ai_runtime.batching import GpuBatchStage
from ai_llm.engine import LLMEngine

logger = logging.getLogger(__name__)


class LLMGPUWorker:
    """Generates answers on the process's GPUs."""

    def __init__(self, config, n_gpus, request_queue, shutdown_event) -> None:
        self._config = config
        # The generation model, spanning every GPU the job owns.
        self._engine = LLMEngine(config.LLM_MODEL, n_gpus,
                                 config.LLM_MAX_MODEL_LEN,
                                 config.LLM_GEN_GPU_MEM_UTIL)
        self._stage = GpuBatchStage(
            self, request_queue, shutdown_event,
            window_size=config.LLM_GPU_SORT_WINDOW_SIZE,
            idle_timeout_s=config.LLM_GPU_BATCH_TIMEOUT_MS / 1000.0,
            max_weight=config.LLM_MAX_TOKENS_PER_BATCH,
            name=type(self).__name__,
        )
        # Set once the engine can generate; loading it takes minutes.
        self.ready = self._stage.ready

    def run(self) -> None:
        """Load the engine, then batch and generate until shutdown."""
        self._stage.run()

    def load(self) -> None:
        """Load the engine onto the GPUs."""
        self._engine.load()

    def unload(self) -> None:
        """Give the GPUs back — the stage on its way out, or the owner."""
        self._engine.unload()

    @staticmethod
    def weight(item) -> int:
        """Prompt length in tokens."""
        return len(item)

    def process(self, items: list, budgets: list) -> list:
        """Return one answer per prompt, each under its own reply budget."""
        logger.info("generating %d answers...", len(items))
        answers = self._engine.generate(items, budgets,
                                        self._config.LLM_TEMPERATURE,
                                        self._config.LLM_LOGPROBS_TOPK)
        logger.info("generated %d answers.", len(answers))
        return answers
