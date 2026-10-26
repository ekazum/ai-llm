# ai-llm

LLM generation service for ML pipelines, built on
[ai-runtime](https://github.com/ekazum/ai-runtime). `[vllm]` adds the vLLM engine.

## Parts

- **Config** (`ai_llm.config.LLMBaseConfig`): model, generation, GPU batching and ports, loaded from YAML; a
  project subclasses it with its own fields.
- **Engine** (`ai_llm.engine.LLMEngine`): one vLLM engine — load tensor-parallel, `chat`, `token_counts`, `generate` from token
  ids under per-prompt budgets with logprob-derived probability values (leading `0`/`1` and decimals), and `unload`.
- **GPU process** (`LLMGPUService`, `LLMGPUProcessor`, `LLMGPUWorker`): processors tokenize under the chat
  template and the worker generates in weight-budgeted batches through ai-runtime's `GpuBatchStage`; `run` keeps
  a socket to the serving task and redials when it goes.
- **Server** (`ai_llm.server.LLMServer`): the serving task's listener — GPU tasks and consumers connect on their
  own ports; prompts go to the GPU task with the smallest backlog, answers back to whoever asked.
- **Serving** (`LLMService`, `LLMProcessor`, `LLMCollector`): the serving manager with its own engine; the
  processor answers from the cache or routes a miss to the GPU, the collector keeps finished answers and sends
  them back.
- **Cache** (`AnswerCache`): a protocol — `lookup(key)`, `update(key, answer)`, `flush()`. The key is any tuple
  the project chooses and the project supplies the implementation; a LangChain `BaseCache` fits behind a short
  adapter.
- **Cache stores** (`CacheSchema`, `ShardCache`, `SqliteAnswerCache`, `CacheBuilder`): one implementation of the
  protocol over integer keys and `(text, *floats)` answers. `ShardCache` keeps a run's answers and flushes them as
  npy shards, `CacheBuilder` folds shards into one SQLite file, and `SqliteAnswerCache` reads that file without
  locking. The schema names the table, the columns and the shard prefixes.
- **Clients** (`LLMBaseClient`, `LLMCachedClient`, `ServicePromptConsumer`, `PromptConsumer`): send prompts,
  receive answers in completion order; `LLMCachedClient` answers from its own cache before asking the service;
  prompt in, text out, with a timeout on a silent service.
- **Stats** (`LLMStats`, `LLMStatsStore`): counters drained by one thread into a shared SQLite store, one row
  set per serving job.

Every part imports from the top level: `from ai_llm import LLMService, AnswerCache, ...`. The project supplies
the question, the cache and the key; `AGENTS.md` lists the queues, signatures, assembly order and the rules to
keep.

## Example

```bash
python examples/fake_engine/service.py
```

The server, consumer and stats with an echo model in a stand-in GPU task; CPU only, no model download.

## License

Apache-2.0
