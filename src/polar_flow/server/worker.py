# server/worker.py
from __future__ import annotations

from pathlib import Path

from .config import Config
from .db import create_session_factory
from .scheduler import scheduler_loop


def run_worker(config_path: str | None = None) -> None:
    cfg = Config.load(Path(config_path)) if config_path else Config.load(Path("config.toml"))
    poll_interval = cfg.server.scheduler_poll_interval
    SessionLocal, _engine = create_session_factory(cfg.server.database_url)  # NEW
    scheduler_loop(poll_interval=poll_interval, session_local=SessionLocal)  # pass in
