"""Migration runner.

Numbered SQL files under forgesight/db/migrations, applied in order inside one
transaction, with the applied set recorded so a rerun is a no-op. Keeping the
migrations as plain files means the deployed schema is readable without running
any Python.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from forgesight.db.pool import Dialect, PoolLike, ph, translate_ddl

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _ensure_bootstrap(conn, dialect: Dialect) -> None:
    """Placeholder style differs per driver: sqlite3 wants `?`, psycopg `%s`.

    Kept in one place so no other migration helper has to care.
    """
    if dialect is Dialect.SQLITE:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration ("
            "  name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        # Backs the fencing token. A dedicated row keeps allocation to a single
        # statement inside the claim transaction.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
        )
        conn.execute("INSERT OR IGNORE INTO counter(name, value) VALUES ('fencing', 0)")
    else:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration ("
            "  name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )


def applied_migrations(conn) -> set[str]:
    rows = conn.fetchall("SELECT name FROM schema_migration")
    return {r["name"] for r in rows}


MIGRATION_RE = re.compile(r"^\d{4,}_[A-Za-z0-9_.-]+\.sql$")


def _migration_files() -> list[Path]:
    """The numbered migration files, in order, ignoring OS metadata.

    `glob("*.sql")` alone also matches macOS AppleDouble sidecar files
    (`._0001_init.sql`), which are created on any non-HFS filesystem -- a Docker
    bind mount, a network share, an extracted tar. They sort *before* the real
    migration, are binary, and made the API fail to start with a
    UnicodeDecodeError from inside this function.

    So: skip dotfiles, and require the numbered name. A file that is neither is
    almost certainly a typo, and that is worth failing on rather than skipping --
    silently ignoring a migration nobody noticed is the failure mode that leaves
    a schema half-applied.
    """
    files = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name.startswith("."):
            continue
        if not MIGRATION_RE.match(path.name):
            raise ValueError(
                f"{path.name} is not a numbered migration "
                f"(expected NNNN_name.sql); rename it or remove it"
            )
        files.append(path)
    return files


def migrate(pool: PoolLike, verbose: bool = True) -> list[str]:
    """Apply every migration not yet recorded. Returns the names applied."""
    dialect = pool.dialect
    placeholder = ph(dialect)
    done = set()
    with pool.connection() as conn:
        _ensure_bootstrap(conn, dialect)
        done = applied_migrations(conn)

    files = _migration_files()
    newly: list[str] = []
    for path in files:
        if path.name in done:
            continue
        sql = path.read_text()
        if dialect is Dialect.SQLITE:
            # sqlite3's executescript() implicitly commits any open transaction
            # before it runs, so nesting it inside pool.write() deadlocks the
            # COMMIT. Putting the transaction in the script text itself is the
            # supported way to make a multi-statement script atomic.
            sql = translate_ddl(sql)
            script = "BEGIN;\n" + sql + "\n" + (
                "INSERT OR REPLACE INTO schema_migration(name) VALUES ('"
                + path.name
                + "');\nCOMMIT;"
            )
            with pool.connection() as conn:
                _ensure_bootstrap(conn, dialect)
                conn.executescript(script)
        else:
            with pool.write() as conn:
                conn.execute(sql)
                conn.execute(
                    f"INSERT INTO schema_migration(name) VALUES ({placeholder})",
                    (path.name,),
                )
        newly.append(path.name)
        if verbose:
            print(f"applied {path.name}")
    return newly


def reset(pool: PoolLike) -> None:
    """Drop and recreate. Only for tests and `make db-reset`."""
    with pool.write() as conn:
        rows = conn.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%'" if pool.dialect is Dialect.SQLITE else
            "SELECT tablename AS name FROM pg_tables WHERE schemaname='public'"
        )
        for r in rows:
            name = r["name"]
            if name in {"schema_migration", "counter"}:
                continue
            conn.execute(f'DROP TABLE IF EXISTS "{name}" CASCADE')
    migrate(pool, verbose=False)


def main() -> int:
    """`python -m forgesight.db.migrate`, the compose entrypoint.

    Lives here rather than in deploy/ so the compose stack needs no extra COPY:
    this module is already inside the package the image copies.
    """
    from forgesight.db.pool import get_pool
    from forgesight.settings import get_settings

    s = get_settings()
    s.ensure_dirs()
    applied = migrate(get_pool(s.database_url), verbose=True)
    print(f"migrations applied: {len(applied)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
