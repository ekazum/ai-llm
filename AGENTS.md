# ai-llm — notes for AI coding tools

ai-llm gives an LLM generation service: a server that consumers and GPU tasks connect to, a GPU process that batches
prompts through vLLM, clients, and a stats store. A project builds a service by subclassing the config, writing two
threads that move work between the server's queues, and calling the clients.

Install: `pip install "ai-llm[vllm] @ git+https://github.com/ekazum/ai-llm.git@v0.1.4"` (`[vllm]` only where the
engine runs). Everything below imports from the top level: `from ai_llm import ...`.

## What ai-llm provides, what the project writes

| Area | ai-llm provides | The project writes |
|---|---|---|
| Config | `LLMBaseConfig` (model, generation, batching, ports) | a subclass with its own fields |
| Server | `LLMServer` (sockets, per-connection threads, GPU dispatch, response routing) | the router: `request_queue` → `gpu_queue` or a direct response |
| GPU side | `LLMGPUService`, `LLMGPUProcessor`, `LLMGPUWorker`, `LLMEngine` | — |
| Answers back | — | the collector: `done_queue` → one `response_queue` item per key |
| Clients | `LLMBaseClient`, `ServicePromptConsumer`, `PromptConsumer` (protocol) | a `LLMBaseClient` subclass for its own message kinds, if any |
| Stats | `LLMStats`, `LLMStatsStore` | — |

## Queues and messages

`LLMServer(config, request_queue, gpu_queue, done_queue, response_queue, stats=None, ready=None)`:

| Queue | Put by | Item | Taken by |
|---|---|---|---|
| `request_queue` | server, per consumer message | `(key, payload, prompt, max_tokens)`; `("prompt", key, text, n)` gives `(key, None, text, n)`, `("lookup", key)` gives `(key, None, None, None)`, `("submit", key, payload)` gives `(key, payload, None, None)` | the project's router |
| `gpu_queue` | the router | `(key, prompt, max_tokens)` or `(key, prompt)` | server, sent to the GPU task with the smallest backlog |
| `done_queue` | server, per GPU message `("done", keys, answers)` | `(keys, answers)` | the project's collector |
| `response_queue` | router (direct answers), collector | `(key, answer)`; `answer` is `(text, *values)` or `None` | server, sent to the consumer that asked for `key` |

Every connection starts with `("hello", "consumer")` or `("hello", "gpu")`. A GPU task may also send
`("stats", source, counters)`. The client port opens only after `ready` is set; the GPU port opens at once. Ports are
`LLM_CLIENT_PORT` / `LLM_GPU_PORT` + `SLURM_JOB_ID % 1000` (`config.client_port(job)`, `config.gpu_port(job)`).

## Signatures

```python
LLMBaseConfig.from_yaml(path)            # YamlConfig: frozen, extra fields forbidden
    .client_port(job_id=None) .gpu_port(job_id=None)
LLMServer(...).run()                     # blocks forever; run it on a daemon thread
LLMGPUService(config, host=None, n_gpus=None, server_job=None, port=None)
    .run()                               # load the engine, dial host:gpu_port, serve, redial on loss
    .start() -> (in_queue, out_queue)  .wait_ready(timeout)  .stop()  .dead_threads()
LLMEngine(model, n_gpus=1, max_model_len=16384, gpu_mem_util=0.9, n_numbers=2)
    .load() .chat(prompts, max_tokens, temperature=0.0) .token_counts(texts) .generate(token_ids, budgets, temperature, logprobs) .unload()
LLMBaseClient(config, host, server_job=None)
    .submit_prompt(key, prompt, max_tokens) .receive() -> (key, answer) .poll(timeout) .close()
ServicePromptConsumer(config, host, server_job=None)        # meets PromptConsumer
    .generate_from_prompt(key, prompt, max_tokens=None) -> str
    .generate_batch_from_prompts(items, max_tokens=None) -> Iterator[list[(key, text)]]
    .answer_text(key) .close()
LLMStats(path, write_every=50)           # .run() on a thread; .event(**counters) .gpu_event(source, counters) .stop() .write()
LLMStatsStore(stats_dir)                 # llm_service.db under stats_dir
```

## Assembly

The serving process (see `examples/fake_engine/service.py`):

1. `config = MyConfig.from_yaml(path)`; four `queue.Queue()`s; `ready = threading.Event()`.
2. `stats = LLMStats(path)`; `server = LLMServer(config, request_queue, gpu_queue, done_queue, response_queue, stats,
   ready=ready)`.
3. Start daemon threads: `server.run`, `stats.run`, the router, the collector.
4. Start the GPU side: `LLMGPUService(config, host="127.0.0.1").run` on a thread in the same job, or as its own job
   with `host` = the server's host and `server_job` = the server's job id. `wait_ready(timeout)` returns once the
   engine has loaded.
5. `ready.set()` opens the client port.
6. Consumers: `ServicePromptConsumer(config, host).generate_batch_from_prompts(items)`.

## Rules that are easy to break

- Keys are tuples, and a consumer receives only the keys it sent; two consumers asking the same key each get one answer.
- An answer is `(text, *values)`; `ServicePromptConsumer` returns `answer[0]`. The collector passes it through.
- The router answers everything that is not a prompt; an unanswered key leaves its consumer waiting until
  `GENERATE_TIMEOUT` (900 s).
- Set `ready` after the work a consumer depends on is done; consumers block until the client port opens.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before starting `LLMGPUService` in a process that may not have touched CUDA.
- `LLMGPUProcessor` loads the tokenizer for `LLM_MODEL` from HuggingFace; a gated model needs accepted terms and a
  token.
- `LLMBaseConfig` subclasses give new field validators names the base does not use (`_non_empty`, `_positive`,
  `_non_negative`, `_fraction`); the same name replaces the base validator.

## Examples

- `examples/fake_engine/service.py` — the server, the router and collector, stats, and a consumer, with an echo model
  in a stand-in GPU task; CPU only, no model download. `python examples/fake_engine/service.py`.

## The five repos

| Repo | Gives | Depends on |
|---|---|---|
| [ai-runtime](https://github.com/ekazum/ai-runtime) | threads, `SHUTDOWN_SENTINEL`, `ThreadStages`, GPU batching, service supervision, SQLite stores, checkpoint managers, `YamlConfig`, logging | standard library; `[torch]`, `[config]` |
| ai-llm | the LLM generation service above | ai-runtime |
| [ai-ui](https://github.com/ekazum/ai-ui) | a Flask web shell and a runs dashboard over a SQLite runs DB | ai-runtime |
| [rl-core](https://github.com/ekazum/rl-core) | RL threads and data types: PPO, prioritized replay, Q-network inference, DQN learner, checkpoint writer | ai-runtime |
| [ai-pipeline](https://github.com/ekazum/ai-pipeline) | data preparation stages | ai-runtime; `[embed]`, `[llm]`, `[reduce]` |
