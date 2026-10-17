"""``LLMBaseConfig`` — the settings the generation service reads.

Model, generation, GPU batching and ports; a project subclasses it with
its own fields.
"""

import os

from pydantic import field_validator

from ai_runtime.yaml_config import YamlConfig


class LLMBaseConfig(YamlConfig):
    """Generation service settings."""

    # LLM id (HuggingFace Hub; gated — needs accepted terms + auth).
    LLM_MODEL: str
    # Cap on generated answer length.
    LLM_MAX_TOKENS: int = 256
    # Top-k token logprobs per generated token.
    LLM_LOGPROBS_TOPK: int = 20
    # Items a GPU worker accumulates before sorting them into batches.
    LLM_GPU_SORT_WINDOW_SIZE: int = 4096
    # Tokens per model call; batches are cut to stay within it.
    LLM_MAX_TOKENS_PER_BATCH: int = 500000
    # Producer-idle wait before a partial window is processed.
    LLM_GPU_BATCH_TIMEOUT_MS: int = 200
    # Threads driving prompts through generation and embedding.
    LLM_GPU_PROCESSOR_THREADS: int = 6
    # Items one processor keeps unfinished before taking more work.
    LLM_GPU_MAX_IN_FLIGHT: int = 4000
    # 0 = deterministic greedy decoding.
    LLM_TEMPERATURE: float = 0.0
    # Per-request ceiling on context + question + generated answer.
    LLM_MAX_MODEL_LEN: int = 40960
    # Fraction of GPU memory for the generation model plus its KV cache.
    LLM_GEN_GPU_MEM_UTIL: float = 0.80
    # Client port base + server job id % 1000 (opened once the server is ready).
    LLM_CLIENT_PORT: int = 5555
    # GPU-task port base + server job id % 1000.
    LLM_GPU_PORT: int = 5556

    @field_validator("LLM_MODEL")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("LLM_MAX_TOKENS", "LLM_LOGPROBS_TOPK", "LLM_MAX_MODEL_LEN",
                     "LLM_GPU_SORT_WINDOW_SIZE", "LLM_MAX_TOKENS_PER_BATCH",
                     "LLM_GPU_MAX_IN_FLIGHT")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"must be >= 1; got {value}")
        return value

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"must be >= 0; got {value}")
        return value

    @field_validator("LLM_GEN_GPU_MEM_UTIL")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"must be in (0, 1); got {value}")
        return value

    def gpu_port(self, job_id: "int | str | None" = None) -> int:
        """The run-scoped GPU-task port: base + server job id % 1000.

        Defaults to this process's own job (the server and the embedded
        worker); a remote GPU task passes the server's job id.
        """
        if job_id is None:
            job_id = os.environ.get("SLURM_JOB_ID") or 0
        return int(self.LLM_GPU_PORT) + int(job_id) % 1000

    def client_port(self, job_id: "int | str | None" = None) -> int:
        """The run-scoped client port: base + server job id % 1000.

        Defaults to this process's own job (the server and in-process
        consumers); a remote consumer passes the server's job id.
        """
        if job_id is None:
            job_id = os.environ.get("SLURM_JOB_ID") or 0
        return int(self.LLM_CLIENT_PORT) + int(job_id) % 1000
