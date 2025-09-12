from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_session_factory(database_url: str) -> tuple[Any, Any]:
    engine = create_engine(database_url, future=True)
    sf = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return sf, engine
