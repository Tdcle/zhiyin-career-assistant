"""Authentication helpers: password hashing and JWT token handling."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.config import config
from utils.redis_client import get_redis_client


# Use PBKDF2 as the default algorithm to avoid bcrypt backend/version conflicts.
# Keep a lightweight fallback verifier for legacy bcrypt hashes already in DB.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

try:
    import bcrypt as _bcrypt  # type: ignore
except Exception:
    _bcrypt = None


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    if not password_hash:
        return False

    # Legacy bcrypt hash support.
    if password_hash.startswith("$2"):
        if _bcrypt is None:
            return False
        try:
            return _bcrypt.checkpw(
                plain_password.encode("utf-8"),
                password_hash.encode("utf-8"),
            )
        except Exception:
            return False

    try:
        return pwd_context.verify(plain_password, password_hash)
    except Exception:
        return False


def create_access_token(user_id: str, username: str, expires_minutes: int | None = None) -> str:
    expire_delta = timedelta(minutes=expires_minutes or config.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expire_delta
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def _revoked_token_key(token: str) -> str:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"auth:revoked:{token_hash}"


def revoke_access_token(token: str) -> bool:
    if not token:
        return False
    payload = decode_access_token(token)
    if not payload:
        return False

    exp = payload.get("exp")
    try:
        expires_at = datetime.fromtimestamp(int(exp), tz=timezone.utc)
    except (TypeError, ValueError):
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    ttl_seconds = int((expires_at - datetime.now(timezone.utc)).total_seconds())
    if ttl_seconds <= 0:
        ttl_seconds = 60

    client = get_redis_client()
    if client is None:
        return False

    try:
        client.setex(_revoked_token_key(token), ttl_seconds, "1")
        return True
    except Exception:
        return False


def is_access_token_revoked(token: str) -> bool:
    if not token:
        return False
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(client.exists(_revoked_token_key(token)))
    except Exception:
        return False


def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def resolve_user_from_request(request: Request) -> tuple[str | None, str | None]:
    token = extract_bearer_token(request)
    if token:
        if is_access_token_revoked(token):
            return None, None
        payload = decode_access_token(token)
        if payload and payload.get("sub"):
            return str(payload["sub"]), str(payload.get("username", ""))
        return None, None
    return request.session.get("user_id"), request.session.get("username")


def require_user_id(request: Request) -> str:
    user_id, username = resolve_user_from_request(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    if username:
        request.session["username"] = username
    request.session["user_id"] = user_id
    return user_id
