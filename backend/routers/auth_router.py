"""Authentication router for JWT-based login and registration."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from schemas.user import (
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
    UserInfo,
)
from utils.auth import (
    create_access_token,
    extract_bearer_token,
    require_user_id,
    resolve_user_from_request,
    revoke_access_token,
)
from db import DatabaseManager
from utils.logger import get_logger


router = APIRouter()
db = DatabaseManager()
logger = get_logger("auth_router")


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, request: Request):
    success, payload = db.create_user_with_password(body.username, body.password)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(payload))

    user_id = payload["user_id"]
    username = payload["username"]
    token = create_access_token(user_id=user_id, username=username)
    request.session["user_id"] = user_id
    request.session["username"] = username
    logger.info("user registered: user_id=%s username=%s", user_id, username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserInfo(user_id=user_id, username=username),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    user = db.authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")

    token = create_access_token(user_id=user["user_id"], username=user["username"])
    request.session["user_id"] = user["user_id"]
    request.session["username"] = user["username"]
    logger.info("user login success: user_id=%s username=%s", user["user_id"], user["username"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserInfo(user_id=user["user_id"], username=user["username"]),
    )


@router.get("/me", response_model=UserInfo)
async def me(request: Request):
    user_id = require_user_id(request)
    _, username = resolve_user_from_request(request)
    return UserInfo(user_id=user_id, username=username or user_id)


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request):
    token = extract_bearer_token(request)
    if token:
        revoke_access_token(token)
    request.session.clear()
    return MessageResponse(success=True, message="logged out")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(body: ChangePasswordRequest, request: Request):
    user_id = require_user_id(request)
    success, message = db.change_user_password(
        user_id=user_id,
        current_password=body.current_password,
        new_password=body.new_password,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(message))
    logger.info("password changed: user_id=%s", user_id)
    return MessageResponse(success=True, message="password changed")
