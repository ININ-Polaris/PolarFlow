# server/worker.py

from __future__ import annotations

from pathlib import Path

from .config import Config
from .scheduler import scheduler_loop


def run_worker(config_path: str | None = None) -> None:
    cfg = Config.load(Path(config_path)) if config_path else Config.load(Path("config.toml"))
    poll_interval = cfg.server.scheduler_poll_interval
    scheduler_loop(poll_interval=poll_interval)
