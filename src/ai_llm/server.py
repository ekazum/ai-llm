"""``LLMServer`` — the service thread consumers and GPU tasks connect to.

Listens on ``LLM_CLIENT_PORT`` / ``LLM_GPU_PORT`` via ``multiprocessing.connection.Listener``, which
moves whole Python objects per message. Every peer identifies itself with a
``("hello", role)`` first message. Consumers send
``("prompt", key, prompt, max_tokens)`` to generate from text they built,
or a project's own ``("lookup", key)`` / ``("submit", key, payload)`` for its
request handler, and get ``(key, answer)`` back. GPU tasks get one prompt per message and send
finished batches back. Each connection has its own reader and writer thread;
two more dispatch prompts round-robin and route responses.
"""

import logging
import threading
from collections import deque
from multiprocessing import connection
from multiprocessing.connection import Listener

import queue

from ai_llm.config import LLMBaseConfig

logger = logging.getLogger(__name__)


class LLMServer:
    """Receives requests, sends responses."""

    def __init__(
        self,
        config: LLMBaseConfig,
        request_queue: queue.Queue,
        gpu_queue: queue.Queue,
        done_queue: queue.Queue,
        response_queue: queue.Queue,
        stats=None,
        ready: threading.Event | None = None,
    ) -> None:
        """Bind the listen ports; the client port waits for ``ready``."""
        self._client_port = config.client_port()
        self._gpu_port = config.gpu_port()
        self._ready = ready
        self._stats = stats
        self._request_queue = request_queue
        self._gpu_queue = gpu_queue
        self._done_queue = done_queue
        self._response_queue = response_queue
        # Every asker of a key waits in line; one response serves one asker.
        self._conn_by_key: dict[tuple, deque] = {}
        # One outbound queue and sender thread per consumer connection.
        self._send_queues: dict[connection.Connection, queue.Queue] = {}
        self._gpu_conns: list[connection.Connection] = []
        self._next_gpu = 0
        self._gpu_outstanding: dict[connection.Connection, int] = {}  # prompts sent, not yet returned
        self._lock = threading.Lock()
        self._gpu_available = threading.Event()

    def run(self) -> None:
        """Accept connections and move messages until shutdown.

        The GPU port opens immediately; the client port only once
        ``ready`` is set, so a client that connects finds the service's
        startup work done.
        """
        threading.Thread(target=self._dispatch_gpu,
                         name="llm-server-gpu", daemon=True).start()
        threading.Thread(target=self._send_responses,
                         name="llm-server-send", daemon=True).start()
        threading.Thread(target=self._accept_loop, args=(self._gpu_port,),
                         name="llm-server-gpu-accept", daemon=True).start()
        if self._ready is not None:
            self._ready.wait()
        self._accept_loop(self._client_port)

    def _accept_loop(self, port: int) -> None:
        """Accept peers on one port; a thread per connection."""
        with Listener(("0.0.0.0", port)) as listener:
            logger.info("LLMServer: listening on port %d.", port)
            while True:
                conn = listener.accept()
                threading.Thread(target=self._serve_connection, args=(conn,),
                                 name="llm-server-conn", daemon=True).start()

    def _serve_connection(self, conn: connection.Connection) -> None:
        """Read one peer until it disconnects; route by its hello role."""
        try:
            _, role = conn.recv()
        except (EOFError, OSError):
            conn.close()
            return
        logger.info("LLMServer: %s connected.", role)
        try:
            if role == "gpu":
                self._serve_gpu(conn)
            else:
                self._serve_consumer(conn)
        except (EOFError, OSError):
            pass
        finally:
            self._drop_connection(conn)
            logger.info("LLMServer: %s disconnected.", role)

    def _serve_gpu(self, conn: connection.Connection) -> None:
        """Register the GPU task; queue its finished work and its counters."""
        with self._lock:
            self._gpu_conns.append(conn)
            self._gpu_outstanding[conn] = 0
            self._gpu_available.set()
        while True:
            message = conn.recv()
            if message[0] == "stats":
                _, source, counters = message
                if self._stats is not None:
                    self._stats.gpu_event(source, counters)
                continue
            _, keys, vectors = message
            with self._lock:
                self._gpu_outstanding[conn] = max(
                    0, self._gpu_outstanding.get(conn, 0) - len(keys))
            self._done_queue.put((keys, vectors))

    def _serve_consumer(self, conn: connection.Connection) -> None:
        """Queue the consumer's submits; its responses get their own thread."""
        outbound: queue.Queue = queue.Queue()
        with self._lock:
            self._send_queues[conn] = outbound
        threading.Thread(target=self._send_to, args=(conn, outbound),
                         name="llm-server-send", daemon=True).start()
        while True:
            message = conn.recv()
            # A lookup carries the key alone, a submit the case, a prompt
            # the text to generate from.
            key = message[1]
            if message[0] == "lookup":
                request = (key, None, None, None)
            elif message[0] == "prompt":
                request = (key, None, message[2], message[3])
            else:
                request = (key, message[2], None, None)
            with self._lock:
                self._conn_by_key.setdefault(key, deque()).append(conn)
            self._request_queue.put(request)

    def _send_to(self, conn: connection.Connection,
                 outbound: queue.Queue) -> None:
        """Write one consumer's responses until its connection is dropped."""
        while True:
            item = outbound.get()
            if item is None:
                break
            try:
                conn.send(item)
                if self._stats is not None:
                    self._stats.event(responses_sent=1)
            except (OSError, ValueError):
                if self._stats is not None:
                    self._stats.event(responses_dropped=1)
                logger.warning("LLMServer: send failed for key %s — dropped.",
                               item[0])

    def _drop_connection(self, conn: connection.Connection) -> None:
        """Forget a disconnected peer; its pending keys are dropped."""
        with self._lock:
            if conn in self._gpu_conns:
                self._gpu_conns.remove(conn)
                self._gpu_outstanding.pop(conn, None)
                if not self._gpu_conns:
                    self._gpu_available.clear()
            for key, waiting in list(self._conn_by_key.items()):
                while conn in waiting:
                    waiting.remove(conn)
                if not waiting:
                    del self._conn_by_key[key]
            outbound = self._send_queues.pop(conn, None)
        if outbound is not None:
            outbound.put(None)
        conn.close()

    def _dispatch_gpu(self) -> None:
        """Send prompts to the GPU task with the smallest backlog, forever."""
        while True:
            item = self._gpu_queue.get()
            while True:
                self._gpu_available.wait()
                with self._lock:
                    if not self._gpu_conns:
                        continue
                    conn = min(self._gpu_conns,
                               key=lambda c: self._gpu_outstanding.get(c, 0))
                try:
                    conn.send(item)
                    with self._lock:
                        self._gpu_outstanding[conn] = (
                            self._gpu_outstanding.get(conn, 0) + 1)
                    break
                except (OSError, ValueError):
                    self._drop_connection(conn)

    def _send_responses(self) -> None:
        """Route finished responses to the sender of the submitting peer."""
        while True:
            key, vector = self._response_queue.get()
            conn: connection.Connection | None = None
            outbound: queue.Queue | None = None
            with self._lock:
                waiting = self._conn_by_key.get(key)
                if waiting:
                    conn = waiting.popleft()
                    if not waiting:
                        del self._conn_by_key[key]
                if conn is not None:
                    outbound = self._send_queues.get(conn)
            if outbound is None:
                if self._stats is not None:
                    self._stats.event(responses_dropped=1)
                logger.warning("LLMServer: no consumer for key %s — dropped.",
                               key)
                continue
            outbound.put((key, vector))
