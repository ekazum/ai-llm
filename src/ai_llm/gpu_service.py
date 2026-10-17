"""``LLMGPUService`` — the GPU process: prompts in, answer texts out.

Runs every role as a thread: ``LLM_GPU_PROCESSOR_THREADS`` ×
``LLMGPUProcessor`` and one ``LLMGPUWorker``. ``start`` returns the queues
work flows through; ``run`` adds the socket to the CPU task. Results leave
per finished batch; parsing to payloads happens at the layout-aware
receiving ends, not here.
"""

import logging
import os
import queue

from ai_runtime.supervisor import ServiceSupervisor
from ai_llm.config import LLMBaseConfig
from ai_llm.gpu_processor import LLMGPUProcessor
from ai_llm.gpu_worker import LLMGPUWorker

logger = logging.getLogger(__name__)


class LLMGPUService:
    """Runs the GPU process: processors, generation, socket."""

    # How long to keep dialling the CPU task before giving up. Generous:
    # the host may be a training job handing over to the evaluation that
    # follows it, and an engine loaded here is worth waiting through that.
    _CONNECT_TIMEOUT: float = 900.0

    def __init__(
        self,
        config: LLMBaseConfig,
        host: str | None = None,
        n_gpus: int | None = None,
        server_job: "int | str | None" = None,
        port: int | None = None,
    ) -> None:
        """Store the CPU task address; discover the allocated GPUs.

        ``server_job`` derives the server's run-scoped port; None means
        the server runs in this very job (the embedded case). An explicit
        ``port`` overrides the derivation.
        """
        self._config = config
        # Empty when the caller drives the queues itself instead of serving.
        self._host: str = host or ""
        self._gpu_port = int(port) if port else config.gpu_port(server_job)
        self._n_gpus = n_gpus or len(
            os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        )
        self._supervisor = ServiceSupervisor(type(self).__name__)
        self._shutdown_event = self._supervisor.shutdown_event
        self._worker: LLMGPUWorker | None = None
        # Processors to generator; drained with the other two at shutdown.
        self._gen_queue: queue.Queue = queue.Queue()

    def start(self) -> tuple[queue.Queue, queue.Queue]:
        """Start the workers and processors; return the in and out queues."""
        in_queue: queue.Queue = queue.Queue()
        out_queue: queue.Queue = queue.Queue()
        gen_queue = self._gen_queue

        # A batch already on the GPU finishes on its own clock, so these are
        # joined with a timeout and must not hold the process open past it.
        worker = LLMGPUWorker(self._config, self._n_gpus, gen_queue,
                              self._shutdown_event)
        self._worker = worker
        self._supervisor.add(worker.run, "llm-generate", True, ready=worker.ready)
        for i in range(self._config.LLM_GPU_PROCESSOR_THREADS):
            self._supervisor.add(
                LLMGPUProcessor(self._config, in_queue, gen_queue,
                                out_queue, self._shutdown_event).run,
                f"llm-gpu-processor-{i}", True,
            )
        self._supervisor.start()
        logger.info("LLMGPUService: %d threads started, %d GPUs.",
                    len(self._supervisor), self._n_gpus)
        return in_queue, out_queue

    def wait_ready(self, timeout: float) -> bool:
        """Block until the engine can generate; False if it never got there."""
        if self._worker is None:
            return False
        return self._worker.ready.wait(timeout)

    def stop(self) -> None:
        """Signal shutdown, join the threads, and release the engine.

        The join is bounded, so a generation thread wedged on a batch is
        abandoned rather than holding shutdown open. The engine is a
        subprocess that outlives this one, so releasing it cannot be left
        to a thread that may never get there.
        """
        self._supervisor.stop(join_timeout=30)
        if self._worker is not None:
            self._worker.unload()

    def dead_threads(self) -> list[str]:
        """Names of threads that have exited."""
        return self._supervisor.dead_threads()

    def _connect(self):
        """Dial the CPU task, retrying while it finishes starting up.

        SLURM reports its job RUNNING before the port is bound, so a GPU
        task that starts first would otherwise die on a refused connection.
        """
        return self._supervisor.dial((self._host, self._gpu_port),
                                     self._CONNECT_TIMEOUT)

    def run(self) -> None:
        """Serving: load the engines once, then keep a socket to the CPU task.

        A restarted CPU task costs a redial, not another engine load.
        """
        in_queue, out_queue = self.start()
        try:
            while not self._shutdown_event.is_set():
                conn = self._connect()
                conn.send(("hello", "gpu"))
                logger.info("LLMGPUService: connected to %s:%d.",
                            self._host, self._gpu_port)
                try:
                    self._serve(conn, in_queue, out_queue)
                except (EOFError, OSError):
                    logger.info("LLMGPUService: server gone, redialling.")
                    # Queued work's asker is gone with the server.
                    self._supervisor.drain(in_queue, self._gen_queue, out_queue)
        finally:
            self.stop()

    def _serve(self, conn, in_queue: queue.Queue,
               out_queue: queue.Queue) -> None:
        """Move work over one connection until it closes."""
        source = os.environ.get("SLURM_JOB_ID", "local")
        while not self._shutdown_event.is_set():
            if conn.poll(0.1):
                in_queue.put(conn.recv())
            while True:
                try:
                    keys, answers = out_queue.get_nowait()
                except queue.Empty:
                    break
                conn.send(("done", keys, answers))
                conn.send(("stats", source,
                           {"batches": 1, "answers": len(keys)}))
                logger.info("LLMGPUService: %d finished sent.", len(keys))
            # A requested stop takes the same threads down; only an
            # unasked-for exit is a failure.
            dead = self.dead_threads()
            if dead and not self._shutdown_event.is_set():
                raise RuntimeError(
                    f"LLMGPUService: threads exited: {dead} — aborting."
                )
