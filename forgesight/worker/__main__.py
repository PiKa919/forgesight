"""Worker entrypoint: `python -m forgesight.worker.main --pool torch`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.settings import get_settings
from forgesight.storage.object_store import build_store
from forgesight.vision.types import Pool
from forgesight.worker.main import Worker, install_signal_handlers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, choices=[p.value for p in Pool])
    ap.add_argument("--once", action="store_true", help="drain what is queued, then exit")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    logging.basicConfig(
        level=args.log_level or s.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    s.ensure_dirs()
    pool = get_pool()
    migrate(pool, verbose=False)
    store = build_store(s)

    worker = Worker(s, pool, store, args.pool)
    log = logging.getLogger("forgesight.worker.main")
    log.info(
        "worker %s up: budget %d B, max_batch %d, threads %d",
        worker.worker_id, worker.admission.effective_budget,
        s.max_batch, s.worker_threads,
    )

    async def run() -> None:
        install_signal_handlers(worker)
        try:
            await worker.run(once=args.once)
        finally:
            log.info("worker %s exiting: %s", worker.worker_id, worker.status()["stats"])

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
