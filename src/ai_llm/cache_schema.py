"""``CacheSchema`` — the columns of an answer store, and the SQL over them.

Integer key columns, one text column, nullable float columns, beside a
``meta`` table of name/value rows. The stores read and write only through
the statements built here.
"""

import re
from dataclasses import dataclass

# A bare SQL identifier: the names are placed into the statements as is.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class CacheSchema:
    """One answer table, keyed by its integer columns, plus ``meta``."""

    # Table holding the answers.
    table: str
    # Integer key columns, in key order; together they are the primary key.
    keys: tuple[str, ...]
    # The answer text column.
    text: str
    # Nullable float columns, in answer order; NaN is stored as NULL.
    floats: tuple[str, ...] = ()
    # Shard file name prefixes for the keys, the texts and the floats.
    key_shard: str = "keys_"
    text_shard: str = "answers_"
    float_shard: str = "lp_"

    def __post_init__(self) -> None:
        """Fail on no key, a repeated column or a name SQL cannot take."""
        if not self.keys:
            raise ValueError("CacheSchema: no key columns.")
        names = [self.table, *self.columns]
        bad = [name for name in names if not _IDENTIFIER.match(name)]
        if bad:
            raise ValueError(f"CacheSchema: not SQL identifiers: {bad}.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"CacheSchema: repeated column in {self.columns}.")

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column of the answer table, in table order."""
        return *self.keys, self.text, *self.floats

    def create_sql(self) -> str:
        """``CREATE TABLE IF NOT EXISTS`` for the answer and ``meta`` tables."""
        answer = self._table_sql(
            self.table,
            [(key, "INTEGER", "NOT NULL") for key in self.keys]
            + [(self.text, "TEXT", "NOT NULL")]
            + [(name, "REAL", "") for name in self.floats],
            f"PRIMARY KEY ({', '.join(self.keys)})",
            " WITHOUT ROWID",
        )
        meta = self._table_sql(
            "meta",
            [("name", "TEXT", "PRIMARY KEY"), ("value", "TEXT", "NOT NULL")],
        )
        return f"{answer};\n\n{meta};\n"

    def select_sql(self) -> str:
        """The text and floats of one key."""
        where = " AND ".join(f"{key} = ?" for key in self.keys)
        return (f"SELECT {', '.join((self.text, *self.floats))} "
                f"FROM {self.table} WHERE {where}")

    def insert_sql(self) -> str:
        """One row, left alone when its key is already stored."""
        marks = ", ".join("?" for _ in self.columns)
        return f"INSERT OR IGNORE INTO {self.table} VALUES ({marks})"

    def count_sql(self) -> str:
        """Rows the answer table holds."""
        return f"SELECT count(*) FROM {self.table}"

    @staticmethod
    def _table_sql(name: str, columns: list[tuple[str, str, str]],
                   constraint: str = "", suffix: str = "") -> str:
        """One ``CREATE TABLE``, its columns aligned on name and type."""
        name_width = max(len(column) for column, _, _ in columns)
        type_width = max(len(kind) for _, kind, rule in columns if rule)
        lines = [
            f"{column.ljust(name_width)} {kind.ljust(type_width)} {rule}"
            if rule else f"{column.ljust(name_width)} {kind}"
            for column, kind, rule in columns
        ]
        if constraint:
            lines.append(constraint)
        body = ",\n    ".join(lines)
        return f"CREATE TABLE IF NOT EXISTS {name} (\n    {body}\n){suffix}"
