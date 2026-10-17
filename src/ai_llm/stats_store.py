"""``LLMStatsStore`` — the serving runs' counters, in one shared file.

The service is fleet-level: it outlives any single training or evaluation
run, and several may serve over a project's life. So unlike the per-record
stores, this one accumulates, and every row carries the SLURM job that
produced it.
"""

from pathlib import Path

from ai_runtime.sqlite_store import SqliteStore


class LLMStatsStore(SqliteStore):
    """One store for every serving run's totals, GPU split and series."""

    FILENAME = "llm_service.db"
    SCHEMA_PATH = Path(__file__).with_name("stats_schema.sql")

    def __init__(self, stats_dir: Path) -> None:
        """Open (creating if absent) the store under the stats directory."""
        super().__init__(stats_dir)
        with self._connect() as db:
            db.executescript(self.SCHEMA_PATH.read_text(encoding="utf-8"))

    def start_run(self, job_id: str, label: str, started_at: str) -> None:
        """Record the run; a restart of the same job keeps its first start."""
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO runs (job_id, label, started_at) "
                "VALUES (?,?,?)", (str(job_id), label, started_at))

    def stop_run(self, job_id: str, stopped_at: str) -> None:
        """Stamp the run as ended; a live run leaves this NULL."""
        with self._connect() as db:
            db.execute("UPDATE runs SET stopped_at = ? WHERE job_id = ?",
                       (stopped_at, str(job_id)))

    def set_counters(self, job_id: str, counters: dict) -> None:
        """Rewrite the run's totals."""
        rows = [(str(job_id), str(k), float(v)) for k, v in counters.items()]
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO counters VALUES (?,?,?)", rows)

    def set_gpu_totals(self, job_id: str, gpu: dict) -> None:
        """Rewrite the per-GPU-task split of those totals."""
        rows = [(str(job_id), str(source), str(k), float(v))
                for source, totals in gpu.items()
                for k, v in totals.items()]
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO gpu_totals VALUES (?,?,?,?)", rows)

    def add_series(self, job_id: str, stamp: str, values: dict) -> None:
        """Append one time-series point for the run."""
        self.append("series", [{"job_id": str(job_id), "time": stamp,
                                **values}], key="job_id")

    def run_ids(self) -> list[str]:
        """Every serving run the store holds, newest last."""
        return [r["job_id"] for r in self.read("runs", order_by="started_at")]
