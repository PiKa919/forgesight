"""Authentication and workspace scoping (design §11.2).

The workspace always comes from the token, never from a request body or a query
parameter. There is no code path that accepts a workspace id from the client,
which is what makes cross-workspace access structurally impossible rather than
merely filtered (AT-11).

Tokens are stored as a salted hash, never in plaintext. A leaked database does
not hand over working credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from forgesight.db.pool import PoolLike, ph
from forgesight.ledger.claims import new_id
from forgesight.settings import get_settings

# A process-local pepper. A real deployment supplies FORGESIGHT_TOKEN_PEPPER via
# the environment; without one, tokens still do not match anything guessable,
# but a restart invalidates them, which is why the default is generated.
_PEPPER = os.environ.get("FORGESIGHT_TOKEN_PEPPER") or secrets.token_hex(16)

bearer = HTTPBearer(auto_error=False)

ROLES = ("viewer", "operator", "owner")


@dataclass(frozen=True, slots=True)
class Principal:
    workspace_id: str
    token_id: str
    role: str

    def require(self, *roles: str) -> None:
        if self.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"role {self.role!r} cannot perform this action",
            )


def hash_token(raw: str) -> str:
    """Peppered SHA-256. The pepper is mixed in as a keyed MAC, so a stolen
    digest cannot be brute-forced offline against a guessable token space."""
    mac = hmac.new(_PEPPER.encode(), raw.encode(), hashlib.sha256)
    return mac.hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def mint(workspace_id: str, role: str = "operator", ttl_h: int | None = None) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    raw = new_token()
    s = get_settings()
    ttl = ttl_h if ttl_h is not None else s.sandbox_ttl_hours
    expires = datetime.now(UTC) + timedelta(hours=ttl)
    pool = get_pool_singleton()
    p = ph(pool.dialect)
    with pool.write() as conn:
        conn.execute(
            f"INSERT INTO api_token(id, workspace_id, token_hash, role, expires_at) "
            f"VALUES ({', '.join([p] * 5)})",
            (new_id("tok"), workspace_id, hash_token(raw), role, expires),
        )
    return raw


# -- pool access ------------------------------------------------------------

_pool: PoolLike | None = None


def get_pool_singleton() -> PoolLike:
    """One pool per process. Created lazily so importing the app is side-effect
    free, which keeps the test suite from opening sockets on import."""
    global _pool
    if _pool is None:
        from forgesight.db.pool import get_pool

        _pool = get_pool()
    return _pool


def set_pool(pool: PoolLike) -> None:
    """Inject a pool. Used by tests and by the worker entrypoints."""
    global _pool
    _pool = pool


def current_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    raw = creds.credentials if creds else x_api_key
    if not raw:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal_for(raw)


def principal_for(raw: str) -> Principal:
    pool = get_pool_singleton()
    p = ph(pool.dialect)
    lookup = (
        "SELECT id, workspace_id, role, expires_at, revoked_at FROM api_token "
        f"WHERE token_hash = {p}"
    )
    with pool.connection() as conn:
        row = conn.fetchone(lookup, (hash_token(raw),))
    if row is None or row["revoked_at"] is not None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    if row["expires_at"] is not None:
        exp = row["expires_at"]
        if isinstance(exp, str):
            exp = datetime.fromisoformat(exp)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        if exp <= datetime.now(UTC):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token expired")
    return Principal(
        workspace_id=row["workspace_id"],
        token_id=row["id"],
        role=row["role"],
    )


def viewer(p: Principal = Depends(current_principal)) -> Principal:
    return p


def operator(p: Principal = Depends(current_principal)) -> Principal:
    p.require("operator", "owner")
    return p


def owner(p: Principal = Depends(current_principal)) -> Principal:
    p.require("owner")
    return p
