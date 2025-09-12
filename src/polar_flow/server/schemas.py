from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polar_flow.server.models import Role, TaskStatus  # noqa: TC001

if TYPE_CHECKING:
    import datetime as dt


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6)

    model_config = ConfigDict(extra="forbid")


class UserRead(BaseModel):
    id: int
    username: str
    role: Role
    visible_gpus: list[int] = Field(default_factory=list)
    priority: int

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    command: str = Field(..., min_length=1)
    requested_gpus: str = Field(..., min_length=1)
    working_dir: str = Field(..., min_length=1, max_length=256)
    gpu_memory_limit: int | None = Field(default=None, ge=0)
    priority: int = Field(default=100, ge=0)

    model_config = ConfigDict(extra="forbid")


class TaskRead(BaseModel):
    id: int
    user_id: int
    name: str
    command: str
    requested_gpus: str
    working_dir: str
    gpu_memory_limit: int | None
    priority: int
    status: TaskStatus
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    stdout_log: str | None
    stderr_log: str | None

    model_config = ConfigDict(from_attributes=True)
