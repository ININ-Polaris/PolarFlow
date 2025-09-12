from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import Config


def main() -> None:
    cfg = Config.load(Path(__file__).parent.parent.parent.parent / "data/config.toml")
    engine = create_engine(cfg.server.database_url, future=True)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
