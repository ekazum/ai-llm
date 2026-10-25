"""``SqliteAnswerCache`` — a built answer store, read only.

The SQLite file a ``CacheBuilder`` wrote, opened ``immutable=1`` so any
number of processes may read it without locking. Its ``meta`` rows must
equal the ones the project expects, or the store is refused.
"""

import logging
import math
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .answer_cache import AnswerCache
from .cache_schema import CacheSchema

logger = logging.getLogger(__name__)


class SqliteAnswerCache(AnswerCache):
    """Answers the build folded in; never written to."""

    def __init__(self, schema: CacheSchema, path: str | Path,
                 expected_meta: dict[str, str]) -> None:
        """Open the store; fail if its ``meta`` differs from ``expected_meta``."""
        self._schema = schema
        self._path = Path(path)
        self._expected_meta = dict(expected_meta)
        self._select = schema.select_sql()
        self._width = len(schema.keys)
        # One handle per reader: sqlite3 forbids two threads driving one.
        self._local = threading.local()
        self._exists = self._open()

    @staticmethod
    def check_store(path: str | Path, expected_meta: dict[str, str]) -> None:
        """Fail loud when the store is absent or its ``meta`` differs —
        before training, not as a stream of misses."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SqliteAnswerCache: {path} does not exist — build it before "
                f"running.")
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            stored = dict(db.execute("SELECT name, value FROM meta").fetchall())
        for name, value in expected_meta.items():
            if stored.get(name) != value:
                raise ValueError(
                    f"SqliteAnswerCache: {path} was built for another {name}.")

    def _open(self) -> bool:
        """Check the store once; absent = every key misses."""
        if not self._path.exists():
            logger.warning("SqliteAnswerCache: %s does not exist — every key "
                           "misses.", self._path)
            return False
        self.check_store(self._path, self._expected_meta)
        logger.info("SqliteAnswerCache: %s opened.", self._path)
        return True

    def _db(self) -> sqlite3.Connection | None:
        """This thread's handle to the store, opened on first use."""
        if not self._exists:
            return None
        if getattr(self._local, "db", None) is None:
            self._local.db = sqlite3.connect(
                f"file:{self._path}?immutable=1", uri=True
            )
        return self._local.db

    def close(self) -> None:
        """Close this thread's handle, if it opened one. Safe to call twice."""
        db: sqlite3.Connection | None = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None

    def lookup(self, key: tuple) -> tuple | None:
        """``AnswerCache.lookup``: ``(text, *floats)``, or ``None``; a key of
        another width is never held."""
        if len(key) != self._width:
            return None
        db = self._db()
        if db is None:
            return None
        try:
            row = db.execute(self._select, key).fetchone()
        except sqlite3.OperationalError:
            # The store was replaced under us — NFS stales cross-node
            # handles on os.replace. Reopen this thread's handle and
            # retry once; the fresh handle reads the new store.
            logger.warning("SqliteAnswerCache: read failed, reopening %s.",
                           self._path)
            self.close()
            db = self._db()
            if db is None:
                return None
            row = db.execute(self._select, key).fetchone()
        if row is None:
            return None
        # NULL (no number in the answer) reads back as nan.
        return row[0], *(math.nan if value is None else value
                         for value in row[1:])

    def update(self, key: tuple, answer: tuple) -> bool:
        """``AnswerCache.update``: the built store is never written here."""
        return False

    def flush(self) -> None:
        """``AnswerCache.flush``: nothing to write."""
