# server/db.py
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(database_url: str) -> tuple[sessionmaker[Session], Engine]:
    """Helper: 创建 SQLAlchemy session 工厂与 engine。
    在 worker 与 app 两边均可重用。
    """
    engine = create_engine(database_url, future=True)
    session_local: sessionmaker[Session] = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    return session_local, engine
