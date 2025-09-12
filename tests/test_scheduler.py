import os
import sys

import pytest

from polar_flow.server.auth import _get_session
from polar_flow.server.models import Task, TaskStatus
from polar_flow.server.scheduler import allocate_and_run_task, resources_available


def _fake_gpu_info(num=2, free_bytes=(4 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)):
    infos = []
    for i in range(num):
        infos.append(
            {
                "id": i,
                "memory_total": 16 * 1024 * 1024 * 1024,
                "memory_free": free_bytes[i],
                "memory_used": 0,
                "util_gpu": 0,
                "util_mem": 0,
            },
        )
    return infos


@pytest.fixture
def mempatch(monkeypatch):
    import polar_flow.server.scheduler as sched

    monkeypatch.setattr(sched, "get_all_gpu_info", lambda: _fake_gpu_info())
    return monkeypatch


def _mk_task(session, user, **kw):
    t = Task(
        user_id=user.id,
        name=kw.get("name", "t"),
        command=kw.get("command", "python -c \"print('ok')\""),
        requested_gpus=kw.get("requested_gpus", "0"),
        gpu_memory_limit=kw.get("gpu_memory_limit"),  # MB or None
        priority=kw.get("priority", 100),
        working_dir=kw.get("working_dir", os.getcwd()),
    )
    session.add(t)
    session.commit()
    return t


def test_resources_available_unit_bug(mempatch):
    # GPU 0 has 4 GB free. Asking for 8000 MB should be NOT available,
    # but current code compares bytes < MB and returns True (bug).
    assert resources_available([0], gpu_memory_limit=8000) is False  # EXPECTED after fix


def test_resources_available_unit_bug_current_behavior(mempatch):
    assert resources_available([0], gpu_memory_limit=8000) is False


def test_allocate_denied_by_visibility(db_session, normal_user, mempatch, tmp_path):
    # user only sees GPU 0; request GPU 1 explicitly
    t = _mk_task(db_session, normal_user, requested_gpus="1", working_dir=tmp_path.as_posix())
    ok = allocate_and_run_task(t, session_local=lambda: _get_session())
    assert ok is False
    # Should still be PENDING
    db_session.refresh(t)
    assert t.status == TaskStatus.PENDING


def test_allocate_and_run_success(db_session, admin_user, mempatch, tmp_path):
    # simple python command that prints to stdout
    cmd = f"{sys.executable} -c \"print('hello')\""
    t = _mk_task(
        db_session,
        admin_user,
        requested_gpus="0",
        command=cmd,
        working_dir=tmp_path.as_posix(),
    )
    ok = allocate_and_run_task(t, session_local=lambda: _get_session())
    assert ok is True
    db_session.refresh(t)
    assert t.status == TaskStatus.SUCCESS
    assert "hello" in (t.stdout_log or "")


def test_allocate_and_run_failure_status(db_session, admin_user, mempatch, tmp_path):
    # non-existent command -> shell returns non-zero
    t = _mk_task(
        db_session,
        admin_user,
        requested_gpus="0",
        command="definitely_not_a_command_xyz",
        working_dir=tmp_path.as_posix(),
    )
    ok = allocate_and_run_task(t, session_local=lambda: _get_session())
    assert ok is True
    db_session.refresh(t)
    assert t.status == TaskStatus.FAILED
    assert t.stderr_log is not None
