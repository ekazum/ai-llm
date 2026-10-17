-- The LLM service's own counters, across every serving run.
--
-- Not per record: a service outlives any one training or evaluation run, so
-- rows carry the SLURM job that produced them and the store accumulates.

-- One row per serving run.
CREATE TABLE IF NOT EXISTS runs (
    job_id     TEXT PRIMARY KEY,
    label      TEXT,
    started_at TEXT,
    stopped_at TEXT              -- NULL while the run is still serving
);

-- The run's totals, rewritten as they climb.
CREATE TABLE IF NOT EXISTS counters (
    job_id TEXT NOT NULL,
    key    TEXT NOT NULL,        -- requests | dispatched | cache_hits | ...
    value  REAL,
    PRIMARY KEY (job_id, key)
) WITHOUT ROWID;

-- The same, split by the GPU task that reported it.
CREATE TABLE IF NOT EXISTS gpu_totals (
    job_id TEXT NOT NULL,
    source TEXT NOT NULL,        -- the GPU task's own job id
    key    TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (job_id, source, key)
) WITHOUT ROWID;

-- The throttled time series: what the run was doing, when.
CREATE TABLE IF NOT EXISTS series (
    job_id           TEXT NOT NULL,
    time             TEXT NOT NULL,
    requests         INTEGER,
    dispatched       INTEGER,
    answers_finished INTEGER,
    cache_hits       INTEGER,
    cache_misses     INTEGER,
    cache_writes     INTEGER,
    gpu_answers      INTEGER,
    PRIMARY KEY (job_id, time)
) WITHOUT ROWID;
