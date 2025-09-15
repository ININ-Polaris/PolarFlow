import sys

from polar_flow.server.auth import _get_session
from polar_flow.server.models import Task, TaskStatus
from polar_flow.worker import scheduler as sched


def _fake_infos(frees):
    # 构造 get_all_gpu_info 返回值（单位：字节）
    res = []
    for i, free in enumerate(frees):
        res.append(
            {
                "id": i,
                "memory_total": 16 * 1024**3,
                "memory_free": free,
                "memory_used": 0,
                "util_gpu": 0,
                "util_mem": 0,
            },
        )
    return res


def test_resources_available_device_missing(monkeypatch):
    monkeypatch.setattr(sched, "get_all_gpu_info", lambda: _fake_infos([4 * 1024**3]))
    assert sched.resources_available([99], gpu_memory_limit=100) is False  # GPU 99 不存在


def test_select_gpus_auto_insufficient(monkeypatch, db_session, normal_user, tmp_path):
    # AUTO:2 但只有 1 块满足 → 返回 []
    monkeypatch.setattr(sched, "get_all_gpu_info", lambda: _fake_infos([5 * 1024**3]))  # 只有 id=0
    t = Task(
        user_id=normal_user.id,
        name="t",
        command="echo x",
        requested_gpus="AUTO:2",
        gpu_memory_limit=100,
        priority=100,
        working_dir=tmp_path.as_posix(),
    )
    db_session.add(t)
    db_session.commit()
    # 直接测内部函数：
    from polar_flow.worker.scheduler import _select_gpus

    assert _select_gpus(t) == []


def test_allocate_and_run_task_env_set(monkeypatch, db_session, admin_user, tmp_path):
    # 命令打印 CUDA_VISIBLE_DEVICES，验证 env 正确设置，覆盖 env 行和成功路径
    monkeypatch.setattr(sched, "get_all_gpu_info", lambda: _fake_infos([8 * 1024**3]))
    cmd = f"{sys.executable} -c \"import os;print(os.environ.get('CUDA_VISIBLE_DEVICES'))\""
    t = Task(
        user_id=admin_user.id,
        name="env",
        command=cmd,
        requested_gpus="0",
        gpu_memory_limit=10,
        priority=100,
        working_dir=tmp_path.as_posix(),
    )
    db_session.add(t)
    db_session.commit()
    ok = sched.allocate_and_run_task(t, session_local=lambda: _get_session())
    assert ok is True
    db_session.refresh(t)
    assert t.status == TaskStatus.SUCCESS
    assert (t.stdout_log or "").strip() == "0"


def test_allocate_and_run_task_resource_insufficient(monkeypatch, db_session, admin_user, tmp_path):
    # resources_available=False 时提前返回（覆盖早退分支）
    monkeypatch.setattr(sched, "get_all_gpu_info", lambda: _fake_infos([1 * 1024**3]))
    t = Task(
        user_id=admin_user.id,
        name="insuf",
        command="echo hi",
        requested_gpus="0",
        gpu_memory_limit=8 * 1024,
        priority=100,
        working_dir=tmp_path.as_posix(),
    )  # 要求 8GB
    db_session.add(t)
    db_session.commit()
    ok = sched.allocate_and_run_task(t, session_local=lambda: _get_session())
    assert ok is False
    db_session.refresh(t)
    assert t.status == TaskStatus.PENDING  # 未启动
