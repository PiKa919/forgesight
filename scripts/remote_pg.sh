#!/usr/bin/env bash
# Provision the remote PostgreSQL used by the dual-dialect test suite.
#
# WHY THIS SCRIPT EXISTS
#
# The remote host is a Lightning AI Studio. Everything outside one Studio folder
# is discarded when the machine stops, /var/lib/docker included, so a container
# created by hand is simply gone on the next start. That is not a hypothesis:
# fs-pg was created exactly that way and had vanished by the next session.
#
# So the container must be recreated from this script rather than from memory.
#
# WHY PGDATA IS *NOT* ON THE PERSISTENT FOLDER
#
# This script previously bind-mounted PGDATA into the Studio folder, on the
# reasonable-sounding theory that it was the only durable path. It does not work.
# The Studio folder is a network FUSE mount, and PostgreSQL needs more from a
# filesystem than durable file contents: after a host restart the container
# refused to start with
#
#     FATAL: could not open directory "pg_notify": No such file or directory
#
# The files were mostly still there, which is exactly what made the mistake easy
# to make. An earlier check proved a table survived `docker rm -f` and concluded
# persistence worked -- but that tested file durability, not whether PostgreSQL
# could *start* from the directory, which is the property that matters.
#
# So PGDATA is container-local by default and the database is recreated from
# scratch on every start. That costs nothing: the test suite drops and recreates
# the schema in a fixture, so there is no state worth keeping. Set PG_DIR only on
# a filesystem with real POSIX semantics (a normal ext4/xfs volume), never on a
# FUSE or network mount.
#
# Usage:
#   scripts/remote_pg.sh up      # create/start, idempotent
#   scripts/remote_pg.sh down    # stop and remove the container
#   scripts/remote_pg.sh status  # is it up, and is the schema present
#
# Env:
#   REMOTE       ssh destination   (or SSH_REMOTE)
#   PG_DIR       PGDATA directory  (default: container-local; see above)
#   LOCAL_PORT   local tunnel port (default: 55432)
#   PG_PASSWORD  superuser password (default: forgesight)

set -euo pipefail

PG_IMAGE="${PG_IMAGE:-postgres:17-alpine}"
PG_DB="${PG_DB:-forgesight}"
PG_USER="${PG_USER:-forgesight}"
PG_PASSWORD="${PG_PASSWORD:-forgesight}"
LOCAL_PORT="${LOCAL_PORT:-55432}"
CONTAINER="${CONTAINER:-fs-pg}"

# Empty means container-local storage, recreated on every `up`. Set this only on
# a real local filesystem; a FUSE mount will produce a PGDATA that looks intact
# and will not start.
PG_DIR="${PG_DIR:-}"
REMOTE="${REMOTE:-${SSH_REMOTE:-}}"

if [ -z "$REMOTE" ]; then
  echo "REMOTE is not set. Export SSH_REMOTE=<user>@<host> or pass REMOTE=..." >&2
  echo "The container must live on the remote host, not here." >&2
  exit 2
fi

log() { printf '[remote-pg] %s\n' "$*" >&2; }

cmd_up() {
  if [ -n "$PG_DIR" ]; then
    log "using PGDATA at $PG_DIR (must be a real POSIX filesystem, not FUSE)"
    ssh "$REMOTE" "mkdir -p '$PG_DIR'"
  else
    log "container-local PGDATA; recreated on every start"
  fi

  if ssh "$REMOTE" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    log "$CONTAINER is already running"
  else
    # A container left behind by a previous host is recreated, not reused: its
    # writable layer may hold a PGDATA that no longer matches PG_DIR (or, on a
    # FUSE mount, does not work at all). --restart unless-stopped covers an
    # in-place restart; this covers a reprovision.
    if ssh "$REMOTE" "docker ps -aq -f name='^$CONTAINER\$'" | grep -q .; then
      log "removing a stopped $CONTAINER from a previous host"
      ssh "$REMOTE" "docker rm -f $CONTAINER" >/dev/null
    fi

    log "starting $CONTAINER ($PG_IMAGE)"
    local volume=""
    if [ -n "$PG_DIR" ]; then
      volume="-v '$PG_DIR:/var/lib/postgresql/data'"
    fi
    ssh "$REMOTE" "docker run -d --name $CONTAINER \
      --restart unless-stopped \
      -p 127.0.0.1:5432:5432 \
      -e POSTGRES_USER=$PG_USER \
      -e POSTGRES_PASSWORD=$PG_PASSWORD \
      -e POSTGRES_DB=$PG_DB \
      $volume \
      $PG_IMAGE" >/dev/null
  fi

  log "waiting for readiness"
  ready=0
  for _ in $(seq 1 30); do
    if ssh "$REMOTE" "docker exec $CONTAINER pg_isready -U $PG_USER" >/dev/null 2>&1; then
      log "postgres is accepting connections"
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    # A restart loop here is nearly always an unusable PGDATA, so say so rather
    # than reporting a generic timeout.
    log "postgres never became ready. Last log lines:"
    ssh "$REMOTE" "docker logs --tail 5 $CONTAINER" >&2 || true
    exit 1
  fi

  log "tunnel: 127.0.0.1:$LOCAL_PORT -> remote 5432"
  log "export FORGESIGHT_TEST_PG='postgresql://$PG_USER:$PG_PASSWORD@127.0.0.1:$LOCAL_PORT/$PG_DB'"
}

cmd_down() {
  log "removing $CONTAINER${PG_DIR:+ (PGDATA at $PG_DIR is left in place)}"
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