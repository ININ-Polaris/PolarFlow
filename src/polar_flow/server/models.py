# server/models.py
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any

from flask_login import UserMixin
from sqlalchemy import JSON, DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Role(Enum):
    USER = "user"
    ADMIN = "admin"


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class User(Base, UserMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[Role] = mapped_column(SAEnum(Role), default=Role.USER, nullable=False)
    visible_gpus: Mapped[list[int]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,  # 注意：用可调用对象，避免共享同一个列表
        nullable=False,
    )
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    # 注意：前向引用用字符串，避免静态类型检查报错
    tasks: Mapped[list[Task]] = relationship(
        "Task",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def set_password(self, raw: str) -> None:
        from werkzeug.security import generate_password_hash  # noqa: PLC0415

        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        from werkzeug.security import check_password_hash  # noqa: PLC0415

        return check_password_hash(self.password_hash, raw)

    def get_visible_gpus_list(self) -> list[int]:
        return self.visible_gpus


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    user: Mapped[User] = relationship("User", back_populates="tasks")

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    command: Mapped[str] = mapped_column(String(512), nullable=False)
    requested_gpus: Mapped[str] = mapped_column(String(64), nullable=False)  # "0,1" 或 "AUTO:2"
    gpu_memory_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)  # MB
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    working_dir: Mapped[str] = mapped_column(String(256), nullable=False)

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus),
        default=TaskStatus.PENDING,
        nullable=False,
    )

    # 使用时区感知时间（UTC），并设置 timezone=True
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC),
        nullable=False,
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    stdout_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr_log: Mapped[str | None] = mapped_column(Text, nullable=True)

    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stdout_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stderr_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    docker_image: Mapped[str | None] = mapped_column(String(256), nullable=True)
    docker_args: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    env: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
