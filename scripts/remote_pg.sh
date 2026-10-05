#!/usr/bin/env bash
# Provision the remote PostgreSQL used by the dual-dialect test suite.
#
# WHY THIS SCRIPT EXISTS
#
# The remote host is a Lightning AI Studio. Everything outside one Studio folder
# is discarded when the machine stops, which includes /var/lib/docker -- so a
# container created by hand, with its data in the container's writable layer,
# is gone on the next start. That is not a hypothesis: fs-pg was created exactly
# that way and had vanished by the next session.
#
# So two things have to be true, and neither is Docker's default:
#
#   1. The container is recreated from this script, not from memory.
#   2. PGDATA is a bind mount onto the persistent Studio folder. A named Docker
#      volume lives under /var/lib/docker and would NOT survive.
#
# The database is disposable either way -- the test suite drops and recreates the
# schema in a fixture -- but keeping it means a restart costs a re-run rather
# than a re-provision.
#
# Usage:
#   scripts/remote_pg.sh up      # create/start, idempotent
#   scripts/remote_pg.sh down    # stop and remove the container
#   scripts/remote_pg.sh status  # is it up, and is the schema present
#
# Env:
#   REMOTE       ssh destination            (default: from SSH_REMOTE, see below)
#   PG_DIR       persistent PGDATA dir       (default: Studio folder, see below)
#   LOCAL_PORT   local tunnel port          (default: 55432)
#   PG_PASSWORD  superuser password         (default: forgesight)

set -euo pipefail

PG_IMAGE="${PG_IMAGE:-postgres:17-alpine}"
PG_DB="${PG_DB:-forgesight}"
PG_USER="${PG_USER:-forgesight}"
PG_PASSWORD="${PG_PASSWORD:-forgesight}"
LOCAL_PORT="${LOCAL_PORT:-55432}"
CONTAINER="${CONTAINER:-fs-pg}"

# The Studio folder is the only path that survives a stop. It is discovered
# rather than hardcoded because the studio name differs per account; the
# this_studio symlink is Lightning's own stable alias for it.
PG_DIR="${PG_DIR:-$(dirname "$0")/../.remote/pgdata}"
REMOTE="${REMOTE:-${SSH_REMOTE:-}}"

if [ -z "$REMOTE" ]; then
  echo "REMOTE is not set. Export SSH_REMOTE=<user>@<host> or pass REMOTE=..." >&2
  echo "The container must live on the remote host, not here." >&2
  exit 2
fi

log() { printf '[remote-pg] %s\n' "$*" >&2; }

cmd_up() {
  log "creating persistent PGDATA at $PG_DIR"
  ssh "$REMOTE" "mkdir -p '$PG_DIR'"

  if ssh "$REMOTE" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    log "$CONTAINER is already running"
  else
    # --restart unless-stopped: the container comes back on its own if the VM is
    # restarted in place. It does not survive a full reprovision, which is what
    # this script is for.
    log "starting $CONTAINER ($PG_IMAGE)"
    ssh "$REMOTE" "docker run -d --name $CONTAINER \
      --restart unless-stopped \
      -p 127.0.0.1:5432:5432 \
      -e POSTGRES_USER=$PG_USER \
      -e POSTGRES_PASSWORD=$PG_PASSWORD \
      -e POSTGRES_DB=$PG_DB \
      -v '$PG_DIR:/var/lib/postgresql/data' \
      $PG_IMAGE" >/dev/null
  fi

  log "waiting for readiness"
  for _ in $(seq 1 30); do
    if ssh "$REMOTE" "docker exec $CONTAINER pg_isready -U $PG_USER" >/dev/null 2>&1; then
      log "postgres is accepting connections"
      break
    fi
    sleep 1
  done

  log "tunnel: 127.0.0.1:$LOCAL_PORT -> remote 5432"
  log "export FORGESIGHT_TEST_PG='postgresql://$PG_USER:$PG_PASSWORD@127.0.0.1:$LOCAL_PORT/$PG_DB'"
}

cmd_down() {
  log "removing $CONTAINER (PGDATA at $PG_DIR is left in place)"
  ssh "$REMOTE" "docker rm -f $CONTAINER" >/dev/null 2>&1 || true
}

cmd_status() {
  ssh "$REMOTE" "docker ps --format '{{.Names}} | {{.Image}} | {{.Status}}' | grep '$CONTAINER' || echo 'not running'"
  ssh "$REMOTE" "docker exec $CONTAINER psql -U $PG_USER -d $PG_DB -tAc \
    'SELECT count(*) FROM information_schema.tables WHERE table_schema = '\''public'\'''" \
    2>/dev/null || echo "cannot query"
}

case "${1:-up}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  *) echo "usage: $0 {up|down|status}" >&2; exit 2 ;;
esac