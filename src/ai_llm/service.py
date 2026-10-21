"""``LLMService`` — the serving manager, hosted inside the job's process.

The hosting job drives the lifecycle (``start`` / ``stop``) and owns the
signal handlers; the threads are daemons.

- ``LLMServer``:    consumer connections — requests in, responses out.
- ``LLMProcessor``: cache check, prompt building, routing to the GPU tasks.
- ``LLMCollector``: finished work — cache the answer, hand to the server.

It also runs its own ``LLMGPUService``, so the job consults its own GPUs over
loopback; remote GPU tasks dial the job's run-scoped GPU port and join the
same round-robin.
"""

import logging
import os
import queue
import threading

from ai_runtime import ThreadHandle

from .answer_cache import AnswerCache
from .collector import LLMCollector
from .config import LLMBaseConfig
from .gpu_service import LLMGPUService
from .processor import LLMProcessor
from .server import LLMServer
from .stats import LLMStats

logger = logging.getLogger(__name__)


class LLMService:
    """Constructs and starts the serving threads; the host calls ``start`` / ``stop``."""

    HOST = "127.0.0.1"
    # Generous: a cold NFS read of the weights is minutes, not seconds.
    ENGINE_TIMEOUT = 1800.0
    _STOP_TIMEOUT = 30.0

    def __init__(self, config: LLMBaseConfig, question: str,
                 cache: AnswerCache | None = None, *,
                 n_gpus: "int | None" = None) -> None:
        """Build the role objects and the queues between them.

        The engine runs here, spanning every visible GPU by default.
        """
        self._config = config
        self._gpu = LLMGPUService(config, host=self.HOST, n_gpus=n_gpus)
        self._gpu_thread: ThreadHandle | None = None
        # Bounded: caps the requests outstanding across the whole service.
        self._request_queue: queue.Queue = queue.Queue(
            maxsize=config.LLM_REQUEST_QUEUE_MAXSIZE
        )
        self._gpu_queue: queue.Queue = queue.Queue()
        self._done_queue: queue.Queue = queue.Queue()
        self._response_queue: queue.Queue = queue.Queue()
        self._stats = LLMStats(config.LLM_STATS_PATH)
        self._cache = cache
        self._collector = LLMCollector(
            self._cache, self._done_queue,
            self._response_queue, self._stats,
        )
        # Set once the threads run; gates the client listen port.
        self._ready = threading.Event()
        self._server = LLMServer(
            config, self._request_queue, self._gpu_queue, self._done_queue,
            self._response_queue, self._stats, ready=self._ready,
        )
        self._processor = LLMProcessor(question, self._request_queue)
        self._handles: list[ThreadHandle] = []

    def start(self) -> str:
        """Start the threads and open the ports; return the host to consult."""
        # None of these return, so they must not hold the host's process
        # open once its own threads have been joined.
        daemon = True
        handles = [
            ThreadHandle(self._server.run, "llm-server", daemon),
            ThreadHandle(self._collector.run, "llm-collector", daemon),
            ThreadHandle(self._stats.run, "llm-stats", daemon),
        ]
        # Cache hits are the common case, so several threads share the queue.
        handles += [
            ThreadHandle(
                lambda: self._processor.serve(
                    self._gpu_queue, self._response_queue, self._cache,
                    self._stats,
                ),
                f"llm-processor-{i}", daemon,
            )
            for i in range(self._config.LLM_PROCESSOR_THREADS)
        ]
        for handle in handles:
            handle.start()
        self._handles = handles
        logger.info("LLMService: %d threads started.", len(handles))
        self._ready.set()
        self._start_engine(self._gpu)
        return self.HOST

    def _start_engine(self, gpu: LLMGPUService) -> None:
        """Load this process's engine; block until it can generate.

        A caller that started consulting first would only pile up requests
        nothing can answer yet.
        """
        # vLLM forks its engine unless it sees CUDA already initialised,
        # and a forked child cannot re-initialise it. Whether the owner
        # touched CUDA first is not something this class should depend on.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        thread = ThreadHandle(gpu.run, "llm-gpu-service", True)
        self._gpu_thread = thread
        thread.start()
        logger.info("LLMService: waiting for the engine to load.")
        if not gpu.wait_ready(self.ENGINE_TIMEOUT):
            thread.join(timeout=0)   # re-raises a load failure
            raise RuntimeError(
                f"LLMService: engine not ready after {self.ENGINE_TIMEOUT:.0f}s"
            )
        logger.info("LLMService: engine ready on %s.", self.HOST)

    def stop(self) -> None:
        """Keep the run's answers and counters; the threads are daemons."""
        self._gpu.stop()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=self._STOP_TIMEOUT)
        if self._cache is not None:
            self._cache.flush()
        self._stats.write()
