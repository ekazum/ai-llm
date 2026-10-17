"""``LLMEngine`` — one vLLM engine, loaded and given back.

The lifecycle a caller needs when it drives a model directly rather than
through the service: load it over the GPUs the job holds, generate from
chat prompts, and release the memory before anything else loads. The
release is the part worth having in one place — the engine is a
subprocess that holds its memory whatever this process does.
"""

import logging
import math
import threading
from typing import Any, cast

logger = logging.getLogger(__name__)


class LLMEngine:
    """A model loaded on this process's GPUs, and the way to let it go."""

    _DIGITS = frozenset("0123456789")
    # Decimal places the question asks for; each is one digit-token expectation.
    _DECIMALS = 4

    def __init__(self, model: str, n_gpus: int = 1,
                 max_model_len: int = 16384,
                 gpu_mem_util: float = 0.9, n_numbers: int = 2) -> None:
        """Name the weights, the room they are loaded into, and the numbers
        ``generate`` reads from each answer's logprobs."""
        self._n_numbers = int(n_numbers)
        self._model = model
        self._n_gpus = int(n_gpus)
        self._max_model_len = int(max_model_len)
        self._gpu_mem_util = float(gpu_mem_util)
        self._llm: Any = None
        self._unload_lock = threading.Lock()

    def load(self) -> None:
        """Load the model tensor-parallel over the GPUs."""
        # noinspection PyPackageRequirements
        from vllm import LLM

        self._llm = LLM(
            model=self._model,
            tensor_parallel_size=self._n_gpus,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_mem_util,
            # No load/capture progress bars — they are stderr-only and fill
            # the task's .err file (TQDM_DISABLE cannot reach them: vLLM
            # passes an explicit disable argument).
            use_tqdm_on_load=False,
        )
        logger.info("%s: model loaded, tensor parallel over %d GPUs.",
                    type(self).__name__, self._n_gpus)

    def chat(self, prompts: list[str], max_tokens: int,
             temperature: float = 0.0) -> list[str]:
        """One reply per prompt, generated together under the chat template."""
        # noinspection PyPackageRequirements
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError(f"{type(self).__name__}: model not loaded.")
        logger.info("%s: generating %d replies...", type(self).__name__,
                    len(prompts))
        outputs = self._llm.chat(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"{len(outputs)} replies for {len(prompts)} prompts — aborting."
            )
        return [output.outputs[0].text for output in outputs]

    def _digit_expectation(self, gen: Any, tokenizer: Any) -> tuple[float, ...]:
        """Logprob-derived values of one generated answer, ``n_numbers`` of them.

        The first ``0.``-prefixed numbers in the token stream, in line
        order; each decimal position is the expected digit over the top-k
        alternatives. ``nan`` where a number is absent.
        """
        cls = type(self)
        if gen.logprobs is None:
            return (math.nan,) * self._n_numbers
        ids = list(gen.token_ids)
        pieces = [tokenizer.decode([tid]) for tid in ids]
        numbers = []
        for i in range(len(ids) - 2):
            if pieces[i].strip() == "0" and pieces[i + 1] == ".":
                positions = [j for j in range(i + 2, i + 2 + cls._DECIMALS)
                             if j < len(ids) and pieces[j] in cls._DIGITS]
                if positions:
                    numbers.append(positions)
        values = []
        for positions in numbers[:self._n_numbers]:
            value, scale = 0.0, 0.1
            for pos in positions:
                probs: dict[int, float] = {}
                for tid, lp in gen.logprobs[pos].items():
                    piece = (lp.decoded_token or tokenizer.decode([tid])).strip()
                    if len(piece) == 1 and piece in cls._DIGITS:
                        probs[int(piece)] = (probs.get(int(piece), 0.0)
                                             + math.exp(lp.logprob))
                mass = sum(probs.values())
                if mass > 0:
                    value += scale * sum(d * p for d, p in probs.items()) / mass
                scale /= 10.0
            values.append(value)
        while len(values) < self._n_numbers:
            values.append(math.nan)
        return tuple(values)

    def generate(self, token_ids: list[list[int]],
                 budgets: list[int], temperature: float = 0.0,
                 logprobs: int = 0) -> list[tuple]:
        """One ``(text, *values)`` per pre-tokenised prompt.

        What the serving path submits: the prompts are already token ids,
        and a batch may mix budgets — an account's is far larger than an
        answer's. ``logprobs`` is the top-k kept per token for the
        logprob-derived values; 0 leaves them ``nan``.
        """
        # noinspection PyPackageRequirements
        from vllm import SamplingParams
        # noinspection PyPackageRequirements
        from vllm.inputs import TokensPrompt

        if self._llm is None:
            raise RuntimeError(f"{type(self).__name__}: model not loaded.")
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
        outputs = self._llm.generate(
            prompts,
            [SamplingParams(temperature=temperature, max_tokens=budget,
                            logprobs=logprobs or None, detokenize=False)
             for budget in budgets],
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"{len(outputs)} answers for {len(prompts)} prompts — aborting."
            )
        tokenizer = self._llm.get_tokenizer()
        return [(tokenizer.decode(output.outputs[0].token_ids,
                                  skip_special_tokens=True).strip(),
                 *self._digit_expectation(output.outputs[0], tokenizer))
                for output in outputs]

    def unload(self) -> None:
        """Give the GPUs back. Safe to call twice, and from any thread.

        The engine is a subprocess holding its memory whatever this process
        does, so whoever gets here first releases it.
        """
        with self._unload_lock:
            engine = getattr(self._llm, "llm_engine", None)
            core = getattr(engine, "engine_core", None)
            shutdown = cast(Any, getattr(core, "shutdown", None))
            if callable(shutdown):
                logger.info("%s: shutting the engine down.",
                            type(self).__name__)
                shutdown()
            self._llm = None
