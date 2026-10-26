"""``LLMEngine`` — one vLLM engine, loaded and given back.

The lifecycle a caller needs when it drives a model directly rather than
through the service: load it over the GPUs the job holds, generate from
chat prompts, and release the memory before anything else loads. The
release is the part worth having in one place — the engine is a
subprocess that holds its memory whatever this process does.
"""

import itertools
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

    def token_counts(self, texts: list[str]) -> list[int]:
        """Tokens in each text under the model's tokenizer, special tokens excluded."""
        if self._llm is None:
            raise RuntimeError(f"{type(self).__name__}: model not loaded.")
        encoded = self._llm.get_tokenizer()(texts, add_special_tokens=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _numbers(self, pieces: list[str]) -> list[tuple[int, list[int]]]:
        """``(leading position, decimal positions)`` of the last ``n_numbers``
        numbers: a digit, ``.``, then up to ``_DECIMALS`` digits."""
        cls = type(self)
        numbers = []
        for i in range(len(pieces) - 2):
            if pieces[i].strip() in cls._DIGITS and pieces[i + 1] == ".":
                positions = [j for j in range(i + 2, i + 2 + cls._DECIMALS)
                             if j < len(pieces) and pieces[j] in cls._DIGITS]
                if positions:
                    numbers.append((i, positions))
        return numbers[-self._n_numbers:]

    def _digit_probs(self, alternatives: dict, tokenizer: Any) -> dict[int, float]:
        """Probability per single digit among one position's top-k alternatives."""
        probs: dict[int, float] = {}
        for tid, lp in alternatives.items():
            piece = (lp.decoded_token or tokenizer.decode([tid])).strip()
            if piece in type(self)._DIGITS:
                probs[int(piece)] = probs.get(int(piece), 0.0) + math.exp(lp.logprob)
        return probs

    def _decimals(self, logprobs: list, tokenizer: Any,
                  positions: list[int]) -> float:
        """Expected value of the decimals at ``positions``, each digit over its
        top-k alternatives."""
        value, scale = 0.0, 0.1
        for pos in positions:
            probs = self._digit_probs(logprobs[pos], tokenizer)
            mass = sum(probs.values())
            if mass > 0:
                value += scale * sum(d * p for d, p in probs.items()) / mass
            scale /= 10.0
        return value

    def _zero_branch_sites(self, gen: Any, tokenizer: Any) -> list[int]:
        """Leading positions written as ``1``; their ``0.`` branch is not in the answer."""
        if gen.logprobs is None:
            return []
        pieces = [tokenizer.decode([tid]) for tid in gen.token_ids]
        return [lead for lead, _ in self._numbers(pieces)
                if pieces[lead].strip() == "1"]

    def _digit_expectation(self, gen: Any, tokenizer: Any,
                           zero_branch: "dict[int, float] | None" = None
                           ) -> tuple[float, ...]:
        """Logprob-derived values of one generated answer, ``n_numbers`` of them.

        Value = P(1) + P(0) * d0: P over the leading 0/1, d0 the ``0.`` branch's
        expected decimals — the answer's own, or ``zero_branch[lead]`` for a
        written ``1``. ``nan`` where a number is absent or not 0/1-led.
        """
        if gen.logprobs is None:
            return (math.nan,) * self._n_numbers
        zero_branch = zero_branch or {}
        pieces = [tokenizer.decode([tid]) for tid in gen.token_ids]
        values = []
        for lead, positions in self._numbers(pieces):
            written = pieces[lead].strip()
            if written == "0":
                d0 = self._decimals(gen.logprobs, tokenizer, positions)
            elif written == "1" and lead in zero_branch:
                d0 = zero_branch[lead]
            else:
                values.append(math.nan)
                continue
            probs = self._digit_probs(gen.logprobs[lead], tokenizer)
            p1 = probs.get(1, 0.0) / (probs.get(0, 0.0) + probs.get(1, 0.0))
            values.append(p1 + (1.0 - p1) * d0)
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
        gens = [output.outputs[0] for output in outputs]
        branches = self._zero_branches(token_ids, gens, tokenizer, temperature,
                                       logprobs)
        return [(tokenizer.decode(gen.token_ids, skip_special_tokens=True).strip(),
                 *self._digit_expectation(gen, tokenizer, branch))
                for gen, branch in zip(gens, branches)]

    def _zero_branches(self, token_ids: list[list[int]], gens: list,
                       tokenizer: Any, temperature: float,
                       logprobs: int) -> list[dict[int, float]]:
        """Per answer, d0 of each ``1``-led number, by one more generate call.

        That call forces ``0.`` in place of the written ``1.`` and reads the
        decimals that follow; a ``0`` outside the top-k has no weight, so 0.
        """
        # noinspection PyPackageRequirements
        from vllm import SamplingParams
        # noinspection PyPackageRequirements
        from vllm.inputs import TokensPrompt

        cls = type(self)
        branches: list[dict[int, float]] = [{} for _ in gens]
        prompts, sites = [], []
        for row, gen in enumerate(gens):
            ids = list(gen.token_ids)
            for lead in self._zero_branch_sites(gen, tokenizer):
                zero_piece = tokenizer.decode([ids[lead]]).replace("1", "0")
                zero = [tid for tid in gen.logprobs[lead]
                        if tokenizer.decode([tid]) == zero_piece]
                if not zero:
                    branches[row][lead] = 0.0
                    continue
                prompts.append(TokensPrompt(prompt_token_ids=(
                    token_ids[row] + ids[:lead] + zero[:1] + [ids[lead + 1]])))
                sites.append((row, lead))
        if not prompts:
            return branches
        outputs = self._llm.generate(
            prompts,
            SamplingParams(temperature=temperature, max_tokens=cls._DECIMALS,
                           logprobs=logprobs, detokenize=False),
            use_tqdm=False,
        )
        for (row, lead), output in zip(sites, outputs):
            forced = output.outputs[0]
            pieces = [tokenizer.decode([tid]) for tid in forced.token_ids]
            positions = list(itertools.takewhile(
                lambda k: pieces[k] in cls._DIGITS, range(len(pieces))))
            branches[row][lead] = self._decimals(forced.logprobs, tokenizer,
                                                 positions)
        logger.info("%s: %d 1-led numbers given their 0. branch.",
                    cls.__name__, len(sites))
        return branches

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
