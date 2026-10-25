"""``CacheBuilder`` — folds answer shards into the SQLite store.

The only writer of the store a ``SqliteAnswerCache`` reads; a store holds
one ``meta``, checked on every build. ``run`` builds beside the live file
and replaces it only once the row count proves nothing was lost;
``fold_hot`` adds a serving run's shards in place and then removes them.
"""

import json
import logging
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from .cache_schema import CacheSchema

logger = logging.getLogger(__name__)


class CacheBuilder:
    """Merges answer shards into the store."""

    def __init__(self, schema: CacheSchema, path: str | Path,
                 shard_dirs: list[str | Path], hot_dir: str | Path,
                 expected_meta: dict[str, str],
                 stats_path: str | Path | None = None) -> None:
        """``run`` reads every ``shard_dirs`` entry; ``fold_hot`` only
        ``hot_dir``; ``stats_path`` receives the build's row counts."""
        self._schema = schema
        self._path = Path(path)
        self._shard_dirs = [Path(directory) for directory in shard_dirs]
        self._hot_dir = Path(hot_dir)
        self._expected_meta = dict(expected_meta)
        self._stats_path = None if stats_path is None else Path(stats_path)

    def _stamp_meta(self, db: sqlite3.Connection) -> None:
        """Write the expected ``meta`` rows into a new store."""
        db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                       self._expected_meta.items())

    def _check_meta(self, db: sqlite3.Connection) -> None:
        """Refuse a store built under another ``meta``."""
        stored = dict(db.execute("SELECT name, value FROM meta").fetchall())
        for name, value in self._expected_meta.items():
            if stored.get(name) != value:
                raise ValueError(
                    f"CacheBuilder: {self._path} was built for another "
                    f"{name} — move it aside and build afresh.")

    def run(self) -> None:
        """Add every shard to a copy of the store, then replace it.

        The copy only grows: a stored key is left alone. It replaces the
        original once its rows equal those it started with plus those added.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        working = self._path.with_name(self._path.name + ".tmp")
        before = self._copy_live(working)

        shards = [p for directory in self._shard_dirs
                  for p in sorted(directory.glob(self._shard_glob()))]
        with closing(sqlite3.connect(str(working))) as db, db:
            db.executescript(self._schema.create_sql())
            if before > 0:
                self._check_meta(db)
            else:
                self._stamp_meta(db)
            db.execute("PRAGMA cache_size = -262144")   # 256 MB page cache
            changes_before = db.total_changes
            offered = self._insert_sorted(db, shards)
            inserted = db.total_changes - changes_before
            rows = db.execute(self._schema.count_sql()).fetchone()[0]

        expected = before + inserted
        if rows != expected:
            raise ValueError(
                f"CacheBuilder: {working} holds {rows} rows, expected "
                f"{expected} ({before} carried over + {inserted} inserted) — "
                f"{self._path} left untouched."
            )
        # Atomic: the store is either the old file or the checked new one.
        os.replace(working, self._path)
        logger.info(
            "CacheBuilder: %d offered, %d inserted, %d rows in %s "
            "(was %d).", offered, inserted, rows, self._path, before)
        self._write_stats(offered, rows, before, inserted)

    def _copy_live(self, working: Path) -> int:
        """Copy the live store aside; return the rows the copy holds.

        The copy must hold what the original holds and pass SQLite's own
        integrity check; a first build starts from no file at all.
        """
        if working.exists():
            raise ValueError(
                f"CacheBuilder: {working} already exists — a previous build "
                f"left it behind. Inspect it and move it away by hand.")
        if not self._path.exists():
            logger.info("CacheBuilder: no store yet, building a new one.")
            return 0
        before = self._row_count(self._path)
        logger.info("CacheBuilder: copying %d rows aside to %s...",
                    before, working.name)
        shutil.copyfile(self._path, working)
        copied = self._row_count(working)
        if copied != before:
            raise ValueError(
                f"CacheBuilder: the copy holds {copied} rows against "
                f"{before} in {self._path} — {self._path} left untouched.")
        with sqlite3.connect(f"file:{working}?mode=ro", uri=True) as db:
            status = db.execute("PRAGMA integrity_check").fetchone()[0]
        if status != "ok":
            raise ValueError(
                f"CacheBuilder: the copy failed its integrity check "
                f"({status}) — {self._path} left untouched.")
        logger.info("CacheBuilder: copy verified, %d rows.", copied)
        return before

    def _row_count(self, path: Path) -> int:
        """Rows the store at ``path`` holds."""
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            return db.execute(self._schema.count_sql()).fetchone()[0]

    def fold_hot(self) -> None:
        """Fold the serving-run shards into the store, then remove them."""
        shards = sorted(self._hot_dir.glob(self._shard_glob()))
        if not shards:
            return
        logger.info("CacheBuilder: folding %d hot shards into %s...",
                    len(shards), self._path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        populated = self._path.exists() and self._row_count(self._path) > 0
        with closing(sqlite3.connect(str(self._path))) as db, db:
            db.executescript(self._schema.create_sql())
            if populated:
                self._check_meta(db)
            else:
                self._stamp_meta(db)
            db.execute("PRAGMA cache_size = -262144")   # 256 MB page cache
            total = self._insert_sorted(db, shards)
        # Remove only after the commit above made the rows durable.
        for keys_path in shards:
            keys_path.unlink()
            self._sibling(keys_path, self._schema.text_shard).unlink()
            self._sibling(keys_path, self._schema.float_shard).unlink()
        logger.info("CacheBuilder: %d hot rows folded from %d shards.",
                    total, len(shards))

    def _shard_glob(self) -> str:
        """The keys shards' file pattern."""
        return f"{self._schema.key_shard}shard*.npy"

    def _sibling(self, keys_path: Path, prefix: str) -> Path:
        """The texts or floats shard written beside a keys shard."""
        return keys_path.with_name(
            keys_path.name.replace(self._schema.key_shard, prefix, 1))

    def _write_stats(self, offered: int, rows: int,
                     carried_over: int = 0, inserted: int = 0) -> None:
        """What the store holds after this build."""
        if self._stats_path is None:
            return
        stats = {
            "module": self._stats_path.stem,
            "rows_offered": offered,
            "rows_carried_over": carried_over,
            "rows_inserted": inserted,
            "rows_in_cache": rows,
        }
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    def _insert_sorted(self, db: sqlite3.Connection,
                       shard_paths: list[Path]) -> int:
        """Insert all shards' rows in key order (sequential B-tree pages)."""
        if not shard_paths:
            return 0
        keys_list, text_list, float_list = [], [], []
        for keys_path in shard_paths:
            keys = np.load(keys_path)
            texts = np.load(self._sibling(keys_path, self._schema.text_shard))
            floats = np.load(self._sibling(keys_path,
                                           self._schema.float_shard))
            if len(keys) != len(texts) or len(keys) != len(floats):
                raise ValueError(
                    f"CacheBuilder: {keys_path.name} has {len(keys)} keys "
                    f"for {len(texts)} texts, {len(floats)} floats — "
                    f"aborting.")
            keys_list.append(keys)
            text_list.append(texts)
            float_list.append(floats)
            logger.info("CacheBuilder: loaded %s — %d rows.",
                        keys_path.name, len(keys))
        keys = np.concatenate(keys_list)
        texts = np.concatenate(text_list)
        floats = np.concatenate(float_list)
        # Last key column first: lexsort sorts by its final argument.
        order = np.lexsort(keys.T[::-1])
        keys, texts, floats = keys[order], texts[order], floats[order]
        # nan (no number in the answer) is stored as NULL.
        db.executemany(
            self._schema.insert_sql(),
            ((*(int(k) for k in key), str(text),
              *(None if np.isnan(value) else float(value) for value in row))
             for key, text, row in zip(keys, texts, floats)),
        )
        logger.info("CacheBuilder: %d rows from %d shards (sorted).",
                    len(keys), len(shard_paths))
        return len(keys)
