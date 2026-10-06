# ADR 0002: two database backends, one set of guarantees

- **Status:** accepted
- **Date:** 2026-10-05

## Context

The design chose PostgreSQL for a reason: `SELECT ... FOR UPDATE SKIP LOCKED`
lets many workers claim disjoint rows in one statement, without blocking and
without handing the same row to two of them.

That was the right choice for the deployment engine. It is not sufficient on its
own, and treating it as if it were had consequences worth writing down.

## The finding

At-most-once processing does not come from `SKIP LOCKED`. It comes from two other
places, and both are enforced by the schema rather than by a query strategy:

1. **The fencing-token compare-and-set.** A claim allocates a strictly increasing
   token and stamps it on the rows. Completion matches on
   `WHERE id = ? AND fencing_token = ? AND state = 'running'`. A worker that lost
   its lease matches zero rows, so its write is a no-op rather than a
   corruption.
2. **`UNIQUE (work_item_id, candidate_id)` on `prediction`.** Even if two workers
   somehow both believed they owned an item, the database refuses the second
   result.

`SKIP LOCKED` is a *contention* optimisation. It changes how cheaply N workers
share a queue. It does not change what happens when a worker is paused, loses its
lease, and comes back.

## Decision

The ledger keeps `SKIP LOCKED` on PostgreSQL and falls back to a serialized
`BEGIN IMMEDIATE` transaction on SQLite. The observable guarantees are identical,
because both live in the schema:

| Property | PostgreSQL | SQLite |
|---|---|---|
| Concurrent claims | disjoint rows, non-blocking | serialized by `BEGIN IMMEDIATE` |
| At-most-once completion | fencing compare-and-set | same |
| Duplicate result | `UNIQUE` constraint | same |
| Lease expiry | reaper | same |

The concurrency-sensitive properties are enforced in one place, the ledger, and
the only per-dialect code is the claim statement's locking clause.

## Why this was worth doing

The test suite and local development had to run on every invocation, including in
environments where no database server is reachable. An alternative was to write
the ledger for PostgreSQL and let the ledger tests skip when it was absent, which
means the fencing and recovery guarantees go unverified in exactly the
configuration most people run.

Instead, **every ledger test runs against both backends.** A fencing bug that
only appears on SQLite is still a bug, and this arrangement means it cannot hide.

## Consequences

- A second dialect exists, so the schema has a translation step
  (`db/pool.py:translate_ddl`) and query helpers (`ph`, `sql`). It is about
  50 lines, and the alternative was unverified concurrency code on the common
  path.
- SQLite serialises writers, so it does not scale past a handful of workers.
  That is fine for local development and wrong for production, which is why
  PostgreSQL remains the deployment engine and the compose default.
- The translation is a regex pass over the migration. It is covered by
  `tests/integration/test_ledger.py`, which creates the schema on both and then
  asserts the constraints the ledger depends on are actually enforced. The
  composite foreign keys, and the prediction uniqueness.

## A related bug this arrangement caught

`FOR UPDATE SKIP LOCKED` and `BEGIN IMMEDIATE` are not the only difference.
SQLite's `CURRENT_TIMESTAMP` has **whole-second** resolution, so the first
implementation measured every latency as 0 s, 1 s or 2 s regardless of what
happened. The design's `clock_timestamp()` is correct on PostgreSQL and silently
useless on SQLite.

Timestamps are now supplied by the application, which keeps sub-second
resolution on both. `tests/integration/test_ledger.py` pins it, and asserts that
service time can never exceed queue-inclusive time, the invariant that the
whole latency story depends on.

## The bug the ledger tests could not catch

The ledger was written dialect-aware from the start. The **API route layer was
not**: 26 statements across four route modules were written with literal `?`
placeholders, which SQLite accepts and psycopg does not. Every API test ran on
SQLite, so the only backend the deployment engine actually uses had never
executed those endpoints.

The first symptom under real PostgreSQL was:

```
psycopg.ProgrammingError: the query has 0 placeholders but 2 parameters were passed
```

psycopg counts `?` as zero placeholders, so the error does not name the query or
the statement. It looks like a parameter-count bug rather than a dialect
mismatch.

### Decision

Portable `?` is now the app's SQL style, and `to_postgres_placeholders()` in
`db/pool.py` rewrites it to `%s` once, at the pool boundary. The rewrite is
quote-aware, so a `?` inside a string literal is left alone.

This was chosen over editing 26 call sites by hand, for one reason: a hand-edited
call site is a chance to introduce a subtle parameter-order bug, and there is no
review step that catches a silently shifted argument list. A translation at the
boundary cannot mis-order parameters, and the next route written with `?` is
correct for free.

The alternative, requiring every caller to ask the dialect for its placeholder
as `WorkspaceRepo` does with `self.ph`, puts the burden on every future caller
and relies on them knowing to. Both styles coexist: `ph()` still exists for code
that interpolates, and statements that already emit the right placeholder are
untouched by the rewrite.

`tests/integration/test_api_dialects.py` runs every API endpoint on both
backends. That file exists because a test suite that only exercises one backend
of a two-backend design is not testing the design.

### Consequences

- The jsonb `?` operator and `?` inside SQL literals are not supported by the
  rewrite. Neither is used anywhere in this codebase, and the docstring on
  `to_postgres_placeholders()` says so at the point of use.
- SQLite stays the fast default because it needs no server, but no longer because
  it is the only thing the API is known to work against.
