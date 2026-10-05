# Runbook

Everything needed to run ForgeSight locally, with containers, and to read its
output honestly. No step here requires a paid service, a GPU, or a credential.

## 1. Local, no container

This is the default and it is not a toy path: the full stack runs this way, and
that is how the numbers in the README were produced.

```bash
make install      # uv sync with all extras
make fetch        # pinned weights, every sha256 verified
make export       # ONNX export with recorded provenance
make datasets     # synthetic evaluation sets
```

**`make fetch` and `make export` are not optional.** They populate the model
registry, and with an empty registry `POST /v1/sessions` returns **503** and
names the directories it expected. That refusal is deliberate: an earlier version
minted the session anyway and handed back a workspace with no candidates and no
active release, where every upload is refused and the token explains nothing.

The same applies to the containers: the `models` and `artifacts` volumes are
bind-mounted from the host for exactly this reason. A fresh clone with empty
volumes boots a healthy API that cannot process a page.

### Why the containers use CPU-only torch

On Linux, `uv.lock` resolves torch to **2.14.1+cpu** rather than the PyPI wheel,
which is a CUDA build. `uv lock` does this automatically, so `make install` and
`podman compose build` agree without extra flags.

The reason is measurement integrity rather than disk space. Initialising a CUDA
context reserves host memory that has nothing to do with the model, and the
headline output of this project is a **per-runtime peak RSS figure**. A CUDA
torch in the torch worker would inflate exactly the number the system exists to
measure, while the ONNX worker's figure stayed honest — so the comparison would
have been structurally unfair, in a direction that flattered whichever runtime
got the smaller image.

Measured on the build host:

| image | size | torch | CUDA |
|---|---|---|---|
| `INSTALL_EXTRAS=torch` | 2.1 GB | `2.14.1+cpu` | not compiled in |
| `INSTALL_EXTRAS=ort` | 1.3 GB | absent | n/a |

Both models run on the CPU-only torch build: `heron` in 4550 ms and
`egret-medium` in 1627 ms for a single 640×640 page on 4 threads, at 827 MB peak
RSS.

macOS is unaffected. The marker is `sys_platform == 'linux'`, so `make install`
on the M5 still gets the native PyPI wheel, which is also CPU-only — there is no
CUDA build for macOS.

Then four processes, in four terminals:

```bash
make api          # 127.0.0.1:8000, serves the built UI too
make worker-torch # the PyTorch pool
make worker-ort   # the ONNX Runtime pool
make reaper       # lease expiry, batch rollup, orphan sweep
```

Open <http://127.0.0.1:8000>. The first load creates a session and keeps its
token in `localStorage`, so a reload works. A token that no longer validates is
discarded and a fresh session is minted, once.

### Database

SQLite by default, at `data/live.db`. For PostgreSQL:

```bash
export FORGESIGHT_DATABASE_URL='postgresql://forgesight:pw@127.0.0.1:5432/forgesight'
make migrate && make api
```

PostgreSQL is the deployment engine and the one the ledger is designed for.
SQLite is the development engine; see
[ADR 0002](adr/0002-ledger-dialects.md) for exactly what is and is not
equivalent.

## 2. Containers

```bash
cp .env.example .env      # then edit it: set the token pepper
podman compose up -d --build
podman compose logs -f api
podman compose down
```

`.env` needs `FORGESIGHT_TOKEN_PEPPER` set to a long random value. It is passed
to the API, both workers and the reaper, and it **must** be identical in all of
them: a per-process pepper means a token minted by one is rejected by the next.

The compose file builds the ORT worker from a stage that installs no torch, on
purpose. If torch were present, the ORT pool's peak RSS could include torch's
allocator and the per-runtime memory figures would stop meaning anything.

**Read this before quoting any container number.** A Podman machine is a Linux
VM with its own CPU count and its own memory. A benchmark taken inside these
containers describes that VM, not the host. It is a separate, labelled
experiment — never comparable with a native run.

## 3. Benchmarks

```bash
make bench                  # throughput and latency, Tier 1 matrix
make report                 # the above plus the quality and gate passes
```

The runner **refuses by default** to measure on battery power, with Low Power
Mode on, or when free memory is under 25%. The environment manifest records the
CPU and its core layout, the OS, the power state, memory pressure before and
after, and every library version.

```bash
uv run python scripts/run_bench.py --tier 1 --allow-unfavourable
```

takes a labelled measurement anyway. The report then carries the refusal reason
on its face and says the numbers should not be compared with a nominal run. This
is deliberate: on a machine that cannot be put on AC power, a labelled
measurement is more useful than nothing, and an unlabelled one is worse than
either.

Useful flags: `--pages`, `--trials`, `--cooldown`, `--models`, `--runtimes`,
`--quality`, `--out`.

A configuration whose trial throughput varies by more than 5%, or drifts more
than 10% from the first trial to the last, is marked **unstable** and excluded
from every comparison. The report prints the reasons.

Output lands in `data/reports/`: a Markdown report, its JSON twin, and
`data/reports/raw/` with one JSON file of raw samples per trial. Every number in
the report is read from one of those files.

## 4. Reading a report

Start with **Limits**. It states that the pages are synthetic, that the absolute
mAP is not comparable with published numbers, and that the measurements are valid
only for the host and date in the manifest.

Then check, in this order:

1. **Host stamp** — if the CPU or library versions are not the ones you care
   about, stop.
2. **Stability** — anything marked unstable is excluded from comparison, and the
   report says which configurations were dropped and why.
3. **Comparisons** — a speedup sentence is only emitted between stable
   configurations from the same run with the same env manifest, and it always
   carries its interval. If a comparison is refused, the sentence says so.
4. **Memory model** — read whether the fit is marked usable. On this host it is
   not; see [ADR 0004](adr/0004-memory-model-did-not-reproduce.md).

## 5. Operations

### The remote PostgreSQL

The dual-dialect suite needs a real PostgreSQL. If the host you have is
**ephemeral** — a Lightning AI Studio, a CI sandbox, anything that discards state
when it stops — read this first.

`scripts/remote_pg.sh` is the supported way to provision it, and it is
idempotent: run it again after a restart instead of rebuilding by hand.

```bash
export SSH_REMOTE='<user>@<host>'
export PG_DIR='/persistent/path/on/that/host'
scripts/remote_pg.sh up       # create or start, then wait for readiness
scripts/remote_pg.sh status   # running? how many tables?
scripts/remote_pg.sh down     # remove the container; PGDATA is kept
```

Then open the tunnel and point the suite at it:

```bash
ssh -f -N -L 55432:127.0.0.1:5432 "$SSH_REMOTE"
export FORGESIGHT_TEST_PG='postgresql://forgesight:forgesight@127.0.0.1:55432/forgesight'
make test-all
```

**`PG_DIR` is the part that matters.** Two defaults are wrong on an ephemeral
host, and both fail silently:

- **Data in the container's writable layer.** Recreating the container discards
  the database.
- **A named Docker volume.** A named volume lives under `/var/lib/docker`, which
  is *not* persistent on these hosts. It survives `docker rm`, so it looks
  correct right up until the machine restarts.

The script bind-mounts `PG_DIR` into the container, so the database outlives the
container. It also passes `--restart unless-stopped`, which covers an in-place
restart but not a full reprovision — that is what re-running the script is for.

On a Lightning AI Studio the persistent location is the Studio folder, visible as
a `lightning` FUSE mount:

```bash
$ mount | grep ' type lightning '
lightning on /home/zeus type lightning (rw,relatime)
lightning on /teamspace/studios/this_studio type lightning (rw,relatime)
```

`/teamspace/studios/this_studio` is the one to use; `this_studio` is Lightning's
own stable alias for it, so it does not change with the account. `/home/zeus` is
also persistent but is the shell's home, so it mixes with dotfiles.

`/teamspace/uploads` is mounted **read-only** — do not try to put PGDATA there.

One wrinkle worth knowing: the Studio folder does not preserve unix ownership, so
PGDATA's `postgres:postgres 700` reads back with an unresolved group. The
container still writes correctly, so this is cosmetic, but `ls` from the login
account will show `UNKNOWN` where a group name would normally appear.

### Migrations

Plain numbered SQL, applied in order inside a transaction, recorded in
`schema_migration`. Re-running is a no-op.

```bash
make migrate
```

There is no down-migration. Immutable data is never migrated destructively: a
schema change adds a column or a table, and old rows are left readable.

### Backups

Back up two things, and note that one of them is content-addressed:

```bash
pg_dump "$FORGESIGHT_DATABASE_URL" > backup.sql
tar czf objects.tgz data/objects
```

`data/objects` is keyed by content hash, so the same bytes are stored once
however many workspaces uploaded them. Restoring is a matter of putting both
back; the database references objects by key, never by absolute path.

### Rotating the demo token

```bash
psql "$FORGESIGHT_DATABASE_URL" -c \
  "UPDATE api_token SET revoked_at = now() WHERE revoked_at IS NULL;"
```

Existing tokens stop working immediately. New sessions are unaffected.

### Purging sandbox data

In public mode, sandbox workspaces carry a 24 h TTL. To purge expired ones:

```bash
psql "$FORGESIGHT_DATABASE_URL" -c "DELETE FROM workspace WHERE expires_at < now();"
```

Objects become unreferenced and the reaper deletes them once they pass the grace
period (`orphan_object_grace_h`, 24 h by default). That delay is why the sweep is
safe to run: it only touches keys that are both unreferenced and old.

### When a worker wedges

Nothing is lost. The lease expires, the reaper requeues the item, and another
worker picks it up. The wedged worker is fenced out when it comes back.

```bash
tail -f logs/worker-torch.log     # look for "lease lost"
podman compose restart worker-torch
```

A worker also recycles itself if its RSS exceeds the hard cap: it finishes the
current batch, releases its leases and exits for the supervisor to restart.

## 6. Troubleshooting

**"running on battery power; connect AC to measure"** — the benchmark guard
working as designed. Plug in, or use `--allow-unfavourable` and accept the label.

**"database is locked"** — SQLite with more writers than it can serialise. WAL
and the busy timeout are set on every connection; if this still appears, the
filesystem does not support WAL locking (some network mounts), so use
PostgreSQL.

**401 on every request** — the token pepper differs between processes, or the
database was reset under a live token. `FORGESIGHT_TOKEN_PEPPER` must be set
identically everywhere; the browser recovers on its own by minting a new session.

**"artifact sha mismatch"** — the weights on disk differ from `models.lock.json`.
The load is refused rather than trusted. Re-fetch with `make fetch --force`.

**A batch stays `queued`** — no worker for that pool. The design routes work to a
pool by the candidate's runtime, so a torch candidate needs `make worker-torch`
running. Check `GET /v1/system/status` for queue depth per pool.
