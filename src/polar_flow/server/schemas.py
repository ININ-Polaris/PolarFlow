# server/schemas.py
from __future__ import annotations

import datetime as dt  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field

from polar_flow.server.models import Role, TaskStatus  # noqa: TC001


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

    docker_image: str | None = Field(default=None, max_length=256)
    docker_args: list[str] | None = Field(default=None, max_items=16)  # 例如 ["--ipc=host"]
    env: dict[str, str] | None = Field(default=None)  # 任务级环境变量

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
    stdout_path: str | None
    stderr_path: str | None

    docker_image: str | None
    docker_args: list[str] | None
    env: dict[str, str] | None

    model_config = ConfigDict(from_attributes=True)
