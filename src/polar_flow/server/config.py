# config.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import toml
from pydantic import BaseModel, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    secret_key: str = Field(...)
    database_url: str = Field(...)
    redis_url: str = Field(...)
    scheduler_poll_interval: int = Field(
        ...,
        gt=0,
        description="轮询间隔（秒），必须大于 0",
    )

    @field_validator("scheduler_poll_interval", mode="after")
    @classmethod
    def check_positive_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("scheduler_poll_interval 必须大于 0")
        return v


class DefaultsConfig(BaseModel):
    user_priority: int = Field(
        default=100,
        ge=0,
        description="默认普通用户提交任务可用的最大优先级(>= 0)",
    )

    @field_validator("user_priority", mode="after")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("user_priority 必须大于等于 0")
        return v


class Config(BaseModel):
    server: ServerConfig
    defaults: DefaultsConfig

    @classmethod
    def load(cls, config_path: Path) -> Config:
        data = {}
        if config_path.exists():
            try:
                data = toml.load(config_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"无法加载配置文件 {config_path}: {e}; 将使用默认配置")
        else:
            logger.warning(f"未找到 {config_path}; 将使用默认配置")

        basedir = config_path.parent

        server_data = data.get("server", {})

        secret_key = server_data.get(
            "secret_key",
            os.environ.get("SECRET_KEY", "you-will-never-guess"),
        )

        database_url = server_data.get("database_url") or os.environ.get("DATABASE_URL")
        if not database_url:
            database_url = f"sqlite:///{(basedir / 'app.db').as_posix()}"

        redis_url = server_data.get("redis_url") or os.environ.get(
            "REDIS_URL",
            "redis://localhost:6379/0",
        )

        # 统一解析 scheduler_poll_interval
        spi_raw = server_data.get("scheduler_poll_interval")
        if spi_raw is None:
            spi_raw = os.environ.get("SCHEDULER_POLL_INTERVAL")
        try:
            spi = int(spi_raw) if spi_raw is not None else 5
        except ValueError:
            logger.warning(
                f"scheduler_poll_interval 无法解析为整数: {spi_raw}, 使用默认 5",
            )
            spi = 5

        # 统一解析 user_priority
        up_raw = (data.get("defaults", {}) or {}).get("user_priority")
        try:
            up = int(up_raw) if up_raw is not None else 100
        except ValueError:
            logger.warning(f"user_priority 无法解析为整数: {up_raw}, 使用默认 100")
            up = 100

        try:
            return cls(
                server=ServerConfig(
                    secret_key=secret_key,
                    database_url=database_url,
                    redis_url=redis_url,
                    scheduler_poll_interval=spi,
                ),
                defaults=DefaultsConfig(user_priority=up),
            )
        except ValidationError:
            logger.exception("配置文件解析错误")
            raise
