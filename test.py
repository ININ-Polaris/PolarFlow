#!/usr/bin/env python3
"""
Integration check for the GPU scheduler server.

What it does:
1) Spins up a temp SQLite DB and creates tables.
2) Creates an admin user, tests /auth/login, /me, /healthz via Flask test client.
3) Mocks GPU info and runs a tiny task through allocate_and_run_task(), verifying SUCCESS.

Run:
  python test_server.py
"""

from __future__ import annotations

from enum import Enum
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ---- import your server package modules ----
# If your package name isn't "server", adjust imports accordingly:
from polar_flow.server.app import create_app
from polar_flow.server.auth import set_session_factory
from polar_flow.server.models import Base, Role, Task, TaskStatus, User
from polar_flow.server.scheduler import allocate_and_run_task


# ---------------- Config helpers ----------------
def make_temp_config(tmpdir: Path, db_url: str) -> Path:
    """
    Create a minimal config.toml that points to our temp SQLite DB and sets a short poll interval.
    """
    config_text = f"""
[server]
secret_key = "dev-secret"
database_url = "{db_url}"
redis_url = "redis://localhost:6379/0"
scheduler_poll_interval = 1

[defaults]
user_priority = 100
""".strip()
    cfg_path = tmpdir / "config.toml"
    cfg_path.write_text(config_text, encoding="utf-8")
    return cfg_path


def build_engine_and_session(db_url: str) -> tuple[sessionmaker[Session], object]:
    engine = create_engine(db_url, future=True)
    SessionLocal: sessionmaker[Session] = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
    )
    return SessionLocal, engine


# ---------------- User / Task fixtures ----------------
def ensure_admin(SessionLocal: sessionmaker[Session]) -> User:
    with SessionLocal() as s:
        user = s.query(User).filter(User.username == "admin").first()
        if user is None:
            user = User(username="admin", role=Role.ADMIN, visible_gpus=[0, 1, 2, 3])
            user.set_password("admin123")
            s.add(user)
            s.commit()
        return user


def create_pending_task(SessionLocal: sessionmaker[Session], user: User) -> Task:
    """
    Create a small 'python -c "print(123)"' task bound to GPU 0.
    """
    tiny_cmd = f'"{sys.executable}" -c "print(123)"'
    with SessionLocal() as s:
        task = Task(
            user_id=user.id,
            name="tiny-echo",
            command=tiny_cmd,
            requested_gpus="0",  # will be accepted because we mock GPU 0 availability
            gpu_memory_limit=None,
            priority=100,
            working_dir=os.getcwd(),
            status=TaskStatus.PENDING,
        )
        s.add(task)
        s.commit()
        s.refresh(task)
        return task


# ---------------- API tests ----------------
def test_auth_and_me(app) -> None:
    """
    Use Flask test_client to verify /auth/login, /me and /healthz.
    """
    # Flask provides a convenient test client for JSON APIs. See docs.  # noqa: E265
    # https://flask.palletsprojects.com/en/stable/testing/
    client = app.test_client()

    # healthz
    rv = client.get("/healthz")
    assert rv.status_code == 200, rv.data

    # login
    rv = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert rv.status_code == 200, rv.data
    # cookie-based session kept in test client

    # me
    rv = client.get("/me")
    assert rv.status_code == 200, rv.data
    data = rv.get_json()
    assert data and data.get("username") == "admin", data


# ---------------- Scheduler test ----------------
def test_scheduler_execution(SessionLocal: sessionmaker[Session]) -> None:
    """
    Mock GPU info and run allocate_and_run_task() once.
    Expect task to finish with SUCCESS and stdout '123\\n'.
    """
    admin = ensure_admin(SessionLocal)
    task = create_pending_task(SessionLocal, admin)

    # ---- patch get_all_gpu_info to pretend we have GPU id=0 with large free memory ----
    # For patching techniques and gotchas, see:
    # https://stackoverflow.com/questions/47223143/python-patching-function-defined-in-same-module-of-tested-function
    from polar_flow.server import scheduler as scheduler_mod

    fake_gpus = [{"id": 0, "memory_free": 64_000}]
    with mock.patch.object(scheduler_mod, "get_all_gpu_info", return_value=fake_gpus):
        ok = allocate_and_run_task(task, SessionLocal)
        assert ok, "allocation should succeed with mocked GPU"

    # reload from DB to see final state
    with SessionLocal() as s:
        fresh = s.get(Task, task.id)
        assert fresh is not None
        assert fresh.status == TaskStatus.SUCCESS, f"task finished with {fresh.status}"
        assert (fresh.stdout_log or "").strip() == "123", f"stdout was: {fresh.stdout_log!r}"
        assert fresh.stderr_log in (None, "") or fresh.stderr_log.strip() == "", (
            f"stderr: {fresh.stderr_log!r}"
        )
        assert fresh.started_at is not None and fresh.finished_at is not None, (
            "timestamps should be populated"
        )


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

# ---------------- main runner ----------------
def main() -> int:
    print("==> Spinning up temp environment...")
    with tempfile.TemporaryDirectory(prefix="gpu_sched_test_") as td:
        tmpdir = Path(td)
        db_url = f"sqlite:///{(tmpdir / 'test.db').as_posix()}"
        cfg_path = make_temp_config(tmpdir, db_url)

        # Prepare engine, session factory & create tables
        SessionLocal, engine = build_engine_and_session(db_url)
        Base.metadata.create_all(bind=engine)
        set_session_factory(SessionLocal)  # let auth.py use our session factory

        print("==> Creating Flask app (test mode) ...")
        app = create_app(str(cfg_path))
        app.json_provider_class.default = EnumEncoder().default
        app.testing = True

        # Ensure admin exists for auth test
        ensure_admin(SessionLocal)

        # Run endpoint tests
        print("==> Testing /healthz, /auth/login, /me ...")
        test_auth_and_me(app)
        print("    ✓ API endpoints OK")

        # Run scheduler test
        print("==> Testing scheduler single task execution ...")
        test_scheduler_execution(SessionLocal)
        print("    ✓ Scheduler executed tiny task successfully")

        print("==> ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
