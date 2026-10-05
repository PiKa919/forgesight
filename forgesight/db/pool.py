"""Connection handling and the small dialect shim.

PostgreSQL is the deployment engine and the one the design describes. SQLite is
the local-dev and test engine, because the project's ledger semantics have to be
exercised on every run and cannot depend on a container being available.

The shim is deliberately tiny and explicit. It exists for one reason: the claim
query needs `SELECT ... FOR UPDATE SKIP LOCKED` on PostgreSQL to let concurrent
workers take disjoint rows without blocking, whereas SQLite serializes writers
with `BEGIN IMMEDIATE`. That is a *contention* difference, not a *correctness*
one. At-most-once completion is enforced by the fencing-token compare-and-set
and by the UNIQUE constraint on prediction(work_item_id, candidate_id), and both
are enforced identically on every backend. See docs/adr/0002-ledger-dialects.md.

Anything that would need a real second dialect fails loudly rather than
silently taking a different code path.
"""

from __future__ import annotations

import enum
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from forgesight.settings import get_settings


class Dialect(enum.Enum):
    POSTGRES = "postgres"
    SQLITE = "sqlite"


def detect_dialect(url: str) -> Dialect:
    return Dialect.SQLITE if url.startswith("sqlite") else Dialect.POSTGRES


class Connection(Protocol):
    def execute(self, sql: str, params: tuple | dict = ()) -> Any: ...
    def executescript(self, script: str) -> None: ...
    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict]: ...
    def fetchone(self, sql: str, params: tuple | dict = ()) -> dict | None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class PoolLike(Protocol):
    dialect: Dialect

    def connection(self) -> Any: ...
    def write(self) -> Any: ...
    def close(self) -> None: ...


class _SqliteConn:
    """Uniform cursor-and-dict interface over a raw sqlite3 connection.

    psycopg already offers dict rows, cursors and transactions under those
    names. This adapter gives sqlite3 the same surface so ledger and repository
    code has no per-dialect branches.
    """

    def __init__(self, raw: Any):
        self._raw = raw

    def execute(self, sql: str, params: Any = ()) -> "_SqliteCursor":
        cur = self._raw.execute(sql, _bind(sql, params))
        return _SqliteCursor(cur)

    def fetchall(self, sql: str, params: Any = ()) -> list[dict]:
        return self.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: Any = ()) -> dict | None:
        return self.execute(sql, params).fetchone()

    def executescript(self, script: str) -> None:
        """Run a multi-statement script. Migrations are one of these."""
        self._raw.executescript(script)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()


class _SqliteCursor:
    def __init__(self, cur: Any):
        self._cur = cur

    def fetchall(self) -> list[dict]:
        cols = [d[0] for d in self._cur.description or []]
        return [_Row(dict(zip(cols, row, strict=True))) for row in self._cur.fetchall()]

    def fetchone(self) -> dict | None:
        cols = [d[0] for d in self._cur.description or []]
        row = self._cur.fetchone()
        return _Row(dict(zip(cols, row, strict=True))) if row is not None else None

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def __iter__(self):
        cols = [d[0] for d in self._cur.description or []]
        for row in self._cur:
            yield _Row(dict(zip(cols, row, strict=True)))


def _bind(sql: str, params: Any) -> Any:
    """Accept a dict or a positional sequence, in either dialect's style.

    The SQL text already carries the placeholder style, so all that is left is
    to keep the parameter container consistent with it.
    """
    if isinstance(params, dict):
        return params
    return tuple(params)


class _Row(dict):
    """dict with attribute access, so ledger code reads naturally on both."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


# --------------------------------------------------------------------------
# DDL translation
# --------------------------------------------------------------------------

# Ordered, explicit rewrites. Each entry is (pattern, replacement) and the
# translation is covered by tests/integration/test_schema.py, which asserts the
# resulting schema actually enforces the constraints the ledger relies on.
_DDL_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bTIMESTAMPTZ\b"), "TEXT"),
    (re.compile(r"\bTIMESTAMP\b"), "TEXT"),
    (re.compile(r"\bnow\(\)"), "CURRENT_TIMESTAMP"),
    (re.compile(r"\bBOOLEAN\b"), "INTEGER"),
    (re.compile(r"CREATE SEQUENCE \w+;"), ""),
    (re.compile(r"nextval\('fencing_seq'\)"), "(SELECT value FROM counter WHERE name='fencing')"),
]


def translate_ddl(sql: str) -> str:
    out = sql
    for pattern, repl in _DDL_RULES:
        out = pattern.sub(repl, out)
    return out


# --------------------------------------------------------------------------
# Query rendering
# --------------------------------------------------------------------------


def claim_sql(d: Dialect) -> str:
    """Take up to `n` queued items for a pool, oldest first.

    The fencing token is supplied by the caller (see `next_fencing_sql`) rather
    than read from a sequence inside the statement. Both dialects then share one
    code path: allocate a token, then stamp it on the claimed rows. A single
    claim may stamp the same token on several rows, which is correct -- the
    token identifies the claim epoch, and what matters for fencing is that it
    strictly increases between claims.

    PostgreSQL: the CTE locks the chosen rows with SKIP LOCKED, so a second
    worker starting at the same instant skips exactly the rows this one is
    taking and claims a disjoint set, with no blocking and no duplicate.
    SQLite: there is no row lock to skip, so the claim runs inside a BEGIN
    IMMEDIATE transaction that excludes other writers for its duration.
    """
    if d is Dialect.POSTGRES:
        return """
WITH c AS (
    SELECT id FROM work_item
    WHERE state = 'queued' AND pool = %(pool)s
    ORDER BY enqueued_at
    FOR UPDATE SKIP LOCKED
    LIMIT %(n)s
)
UPDATE work_item w
SET state = 'running',
    lease_owner = %(worker)s,
    lease_expires_at = clock_timestamp() + (%(lease_s)s * interval '1 second'),
    fencing_token = %(token)s,
    attempts = w.attempts + 1,
    claimed_at = clock_timestamp()
FROM c
WHERE w.id = c.id AND w.state = 'queued'
RETURNING w.*;
"""
    return """
UPDATE work_item
SET state = 'running',
    lease_owner = :worker,
    lease_expires_at = datetime('now', :lease_s || ' seconds'),
    fencing_token = :token,
    attempts = attempts + 1,
    claimed_at = CURRENT_TIMESTAMP
WHERE id IN (
    SELECT id FROM work_item
    WHERE state = 'queued' AND pool = :pool
    ORDER BY enqueued_at
    LIMIT :n
)
AND state = 'queued'
RETURNING *;
"""


def next_fencing_sql(d: Dialect) -> str:
    if d is Dialect.POSTGRES:
        return "SELECT nextval('fencing_seq') AS token;"
    return "UPDATE counter SET value = value + 1 WHERE name = 'fencing' RETURNING value AS token;"


def now_sql(d: Dialect) -> str:
    return "clock_timestamp()" if d is Dialect.POSTGRES else "CURRENT_TIMESTAMP"


def ph(d: Dialect) -> str:
    """Positional placeholder for the dialect: `?` on sqlite3, `%s` on psycopg.

    Positional rather than named because the two drivers disagree on named
    syntax too (`:name` vs `%(name)s`), which would mean duplicating every SQL
    string. The call sites are few and each passes one parameter.
    """
    return "?" if d is Dialect.SQLITE else "%s"


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------


class SqlitePool:
    """Thread-local SQLite connections with WAL and enforced foreign keys."""

    dialect = Dialect.SQLITE

    def __init__(self, url: str):
        self._path = url.split("sqlite:///", 1)[-1] or url.replace("sqlite://", "")
        self._local = threading.local()
        self._all: list[Any] = []
        self._lock = threading.Lock()
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=10000")

    def _new(self) -> _SqliteConn:
        import sqlite3

        raw = sqlite3.connect(self._path, timeout=10.0, isolation_level=None)
        with self._lock:
            self._all.append(raw)
        return _SqliteConn(raw)

    @property
    def conn(self):
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._new()
            self._local.conn = c
        return c

    @contextmanager
    def connection(self) -> Iterator[Any]:
        yield self.conn

    @contextmanager
    def write(self) -> Iterator[Any]:
        """An exclusive write transaction. Mirrors PostgreSQL's row locking
        for the duration of a claim, at the cost of blocking other writers."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                try:
                    c.close()
                except Exception:
                    pass
            self._all.clear()


class _PsycopgConn:
    """Uniform interface over a psycopg3 connection.

    psycopg3 puts fetchall/fetchone on the cursor, not the connection, and its
    default row factory yields tuples. This adapter gives it the same surface as
    `_SqliteConn` so repository code never branches on the backend.
    """

    def __init__(self, conn: Any):
        self._conn = conn

    def execute(self, sql: str, params: Any = ()) -> Any:
        return self._conn.execute(sql, _bind(sql, params))

    def executescript(self, script: str) -> None:
        self._conn.execute(script)

    def fetchall(self, sql: str, params: Any = ()) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, _bind(sql, params)).fetchall()]

    def fetchone(self, sql: str, params: Any = ()) -> dict | None:
        row = self._conn.execute(sql, _bind(sql, params)).fetchone()
        return dict(row) if row is not None else None

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


class PostgresPool:
    dialect = Dialect.POSTGRES

    def __init__(self, url: str, min_size: int = 1, max_size: int = 8):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            url,
            min_size=min_size,
            max_size=max_size,
            open=True,
            timeout=15,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            yield _PsycopgConn(conn)

    @contextmanager
    def write(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            with conn.transaction():
                yield _PsycopgConn(conn)

    def close(self) -> None:
        self._pool.close()


def get_pool(url: str | None = None):
    url = url or get_settings().database_url
    if detect_dialect(url) is Dialect.SQLITE:
        return SqlitePool(url)
    return PostgresPool(url)
