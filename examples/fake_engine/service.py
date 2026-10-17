"""The generation service end to end on the CPU, with a stand-in for the GPU task.

    python examples/fake_engine/service.py

ai-llm supplies the server, the consumer and the stats; this file writes the
two threads a project adds between them (the router and the collector) and a
GPU task that speaks the server's GPU protocol with an echo model in place of
vLLM. A real deployment runs ``LLMGPUService`` there instead.
"""

import logging
import queue
import tempfile
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

from ai_runtime import BatchRequest, GpuBatchStage, ThreadHandle

from ai_llm import LLMBaseConfig, LLMServer, LLMStats, ServicePromptConsumer

HOST = "127.0.0.1"


class EchoModel:
    """A ``BatchModel``: the answer is the prompt upper-cased, cut to its budget."""

    @staticmethod
    def load() -> None:
        pass

    @staticmethod
    def unload() -> None:
        pass

    @staticmethod
    def weight(item: str) -> int:
        return len(item)

    @staticmethod
    def process(items: list, budgets: list) -> list:
        # One (text, *values) tuple per item, the shape LLMEngine.generate returns.
        return [(item.upper()[:budget], float(len(item))) for item, budget in zip(items, budgets)]


class FakeGPUTask:
    """Dials the server's GPU port; batches prompts through ``GpuBatchStage``."""

    def __init__(self, config: LLMBaseConfig, shutdown: threading.Event) -> None:
        self._port = config.gpu_port()
        self._shutdown = shutdown
        self._requests: queue.Queue = queue.Queue()
        self._finished: queue.Queue = queue.Queue()
        self.stage = GpuBatchStage(EchoModel(), self._requests, shutdown,
                                   window_size=64, idle_timeout_s=0.05, max_weight=200)

    def run(self) -> None:
        """Take ``(key, prompt, budget)`` messages; send back ``("done", keys, answers)``."""
        conn = None
        while conn is None:
            try:
                conn = Client((HOST, self._port))
            except OSError:
                time.sleep(0.1)          # the server is still binding its port
        conn.send(("hello", "gpu"))
        while not self._shutdown.is_set():
            if conn.poll(0.05):
                key, prompt, budget = conn.recv()
                self._requests.put(BatchRequest(keys=[key], items=[prompt],
                                                sink=self._finished, budgets=[budget]))
            keys, answers = [], []
            while not self._finished.empty():
                key, answer = self._finished.get()
                keys.append(key)
                answers.append(answer)
            if keys:
                conn.send(("done", keys, answers))
                conn.send(("stats", "fake-gpu", {"batches": 1, "answers": len(keys)}))
        conn.close()


def route(request_queue: queue.Queue, gpu_queue: queue.Queue,
          response_queue: queue.Queue) -> None:
    """The project's router: a prompt goes to the GPU, anything else is answered empty."""
    while True:
        key, _payload, prompt, budget = request_queue.get()
        if prompt is not None:
            gpu_queue.put((key, prompt, budget))
        else:
            response_queue.put((key, None))


def collect(done_queue: queue.Queue, response_queue: queue.Queue, stats: LLMStats) -> None:
    """The project's collector: one finished batch becomes one response per key."""
    while True:
        keys, answers = done_queue.get()
        for key, answer in zip(keys, answers):
            response_queue.put((key, answer))
        stats.event(answers_finished=len(keys))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    config = LLMBaseConfig(LLM_MODEL="echo", LLM_MAX_TOKENS=32,
                           LLM_CLIENT_PORT=47555, LLM_GPU_PORT=47556)
    request_queue, gpu_queue = queue.Queue(), queue.Queue()
    done_queue, response_queue = queue.Queue(), queue.Queue()
    shutdown, ready = threading.Event(), threading.Event()

    with tempfile.TemporaryDirectory() as tmp:          # the stats store; removed on exit
        stats = LLMStats(str(Path(tmp) / "llm_stats"))
        server = LLMServer(config, request_queue, gpu_queue, done_queue, response_queue,
                           stats, ready=ready)
        gpu = FakeGPUTask(config, shutdown)
        # The server's threads never return; daemons let the process end.
        threads = [ThreadHandle(server.run, "llm-server", daemon=True),
                   ThreadHandle(stats.run, "llm-stats", daemon=True),
                   ThreadHandle(lambda: route(request_queue, gpu_queue, response_queue),
                                "router", daemon=True),
                   ThreadHandle(lambda: collect(done_queue, response_queue, stats),
                                "collector", daemon=True),
                   ThreadHandle(gpu.stage.run, "fake-gpu-stage", daemon=True),
                   ThreadHandle(gpu.run, "fake-gpu", daemon=True)]
        for thread in threads:
            thread.start()
        gpu.stage.ready.wait(5)
        ready.set()                                     # opens the client port

        consumer = ServicePromptConsumer(config, HOST)
        prompts = [((i,), f"prompt number {i}") for i in range(6)]
        for batch in consumer.generate_batch_from_prompts(prompts, max_tokens=8):
            for key, text in batch:
                print(f"{key}: {text!r}")
        print("single:", consumer.generate_from_prompt(("one",), "hello service"))
        consumer.close()

        shutdown.set()
        stats.stop()
        threads[1].join(5)
        print("stats store:", sorted(p.name for p in Path(tmp).iterdir()))


if __name__ == "__main__":
    main()
