# server/worker.py
from __future__ import annotations

from pathlib import Path

from polar_flow.server.config import Config
from polar_flow.server.db import create_session_factory
from polar_flow.server.models import Base
from polar_flow.server.scheduler import scheduler_loop


def run_worker(config_path: str | None = None) -> None:
    cfg = Config.load(Path(config_path) if config_path else Path("config.toml"))
    poll_interval = cfg.server.scheduler_poll_interval
    session_local, engine = create_session_factory(cfg.server.database_url)
    Base.metadata.create_all(engine)  # ensure tables exist
    scheduler_loop(poll_interval=poll_interval, session_local=session_local)

def main() -> None:
    run_worker("data/config.toml")
