from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    import datetime as dt

    from server.models import Role, TaskStatus


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

    model_config = ConfigDict(from_attributes=True)


class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    command: str = Field(..., min_length=1)
    requested_gpus: str = Field(..., min_length=1)
    working_dir: str = Field(..., min_length=1, max_length=256)
    gpu_memory_limit: int | None = Field(default=None, ge=0)
    priority: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("priority", mode="after")
    @classmethod
    def default_priority_if_missing(cls, v: int | None) -> int:
        return 100 if v is None else v


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
