"""Reaper entrypoint: `python -m forgesight.ledger.reaper`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.ledger.reaper import Reaper
from forgesight.settings import get_settings
from forgesight.storage.object_store import build_store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    ap.add_argument("--no-sweep", action="store_true", help="skip the object sweep")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    logging.basicConfig(level=args.log_level or s.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
    log = logging.getLogger("forgesight.reaper")
    s.ensure_dirs()
    pool = get_pool()
    migrate(pool, verbose=False)
    store = None if args.no_sweep else build_store(s)
    reaper = Reaper(pool, s)

    running = True

    def stop(*_a) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while running:
        res = reaper.tick(store)
        if res.requeued or res.exhausted or res.objects_deleted:
            log.info("reaped %s", res)
        if args.once:
            break
        # Sleep in slices so a signal is noticed promptly rather than after a
        # whole interval.
        deadline = time.monotonic() + s.reaper_interval_s
        while running and time.monotonic() < deadline:
            time.sleep(min(0.25, s.reaper_interval_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
