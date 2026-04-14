"""User and auth schemas."""

from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, description="Username")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=6, max_length=128, description="Password")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=6, max_length=128, description="Password")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=128, description="Current password")
    new_password: str = Field(..., min_length=6, max_length=128, description="New password")


class SelectUserRequest(BaseModel):
    username: str = Field(..., description="User to switch")


class UserInfo(BaseModel):
    user_id: str
    username: str


class UserListResponse(BaseModel):
    users: list[str]


class MessageResponse(BaseModel):
    success: bool
    message: str


class MemoryFactResponse(BaseModel):
    id: int
    fact_key: str
    fact_value: str
    source: str
    is_active: bool
    confidence: float
    importance: int
    expires_at: str | None = None
    last_used_at: str | None = None
    use_count: int = 0
    meta: dict = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class MemoryFactListResponse(BaseModel):
    memories: list[MemoryFactResponse]


class UpdateMemoryFactRequest(BaseModel):
    fact_key: str = Field(..., min_length=1, max_length=80)
    fact_value: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    meta: dict | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


class ResumeTaskSubmitResponse(BaseModel):
    success: bool
    task_id: str
    status: str
    message: str


class ResumeTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    filename: str | None = None
    user_id: str | None = None
    updated_at: str | None = None
