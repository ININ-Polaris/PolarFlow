import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from polar_flow.server.app import create_app
from polar_flow.server.auth import _get_session
from polar_flow.server.models import Role, User


@pytest.fixture(scope="session")
def tmp_db_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def config_file(tmp_db_dir):
    db_path = tmp_db_dir / "app.db"
    cfg_path = tmp_db_dir / "config.toml"
    cfg_path.write_text(
        f"""
[server]
secret_key = "test-secret"
database_url = "sqlite:///{db_path.as_posix()}"
redis_url = "redis://localhost:6379/0"
scheduler_poll_interval = 1

[defaults]
user_priority = 100
""".strip(),
    )
    return cfg_path


@pytest.fixture
def app(config_file):
    app = create_app(str(config_file))
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db_session(app) -> Generator[Session, None, None]:
    sess = _get_session()
    try:
        yield sess
    finally:
        sess.close()


@pytest.fixture
def admin_user(db_session) -> User:
    u = db_session.query(User).filter_by(username="admin").first()
    if u is None:
        u = User(username="admin", role=Role.ADMIN, visible_gpus=[0, 1, 2], priority=100)
        u.set_password("secret123")
        db_session.add(u)
        db_session.commit()
    return u


@pytest.fixture
def normal_user(db_session) -> User:
    u = db_session.query(User).filter_by(username="alice").first()
    if u is None:
        u = User(username="alice", role=Role.USER, visible_gpus=[0], priority=100)
        u.set_password("secret123")
        db_session.add(u)
        db_session.commit()
    return u
