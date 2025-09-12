# server/scheduler.py
from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from polar_flow.server.gpu_monitor import get_all_gpu_info
from polar_flow.server.models import Role, Task, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

SessionFactory = Callable[[], Session]


def resources_available(requested: list[int], gpu_memory_limit: int | None) -> bool:
    """
    检查给定 GPU 是否有足够的可用显存。
    NVML 返回的是字节，这里将 gpu_memory_limit(单位 MB) 转换为字节后比较。
    """
    infos = get_all_gpu_info()
    free_map: dict[int, int] = {g["id"]: g["memory_free"] for g in infos}  # bytes

    for gid in requested:
        free_bytes = free_map.get(gid)
        if free_bytes is None:
            return False
        if gpu_memory_limit is not None:
            required_bytes = gpu_memory_limit * 1024 * 1024  # MB -> bytes
            if free_bytes < required_bytes:
                return False
    return True


def _select_gpus(task: Task) -> list[int]:
    if task.requested_gpus.startswith("AUTO:"):
        num = int(task.requested_gpus.split(":", 1)[1])
        infos = get_all_gpu_info()

        # 注意：NVML 是字节，这里做单位换算
        limit_bytes = None
        if task.gpu_memory_limit is not None:
            limit_bytes = task.gpu_memory_limit * 1024 * 1024

        candidates = [g for g in infos if (limit_bytes is None or g["memory_free"] >= limit_bytes)]
        if len(candidates) < num:
            return []
        selected = [
            g["id"] for g in sorted(candidates, key=lambda x: x["memory_free"], reverse=True)[:num]
        ]
    else:
        selected = [int(x) for x in task.requested_gpus.split(",") if x.strip() != ""]
    return selected


def allocate_and_run_task(task: Task, session_local: SessionFactory) -> bool:
    session: Session = session_local()
    try:
        # 在当前 session 中把 task 捞出来（顺便把 user 一并 eager load，避免再次懒加载）
        task_db = session.execute(
            select(Task).options(joinedload(Task.user)).where(Task.id == task.id)
        ).scalar_one_or_none()
        if task_db is None:
            return False

        selected = _select_gpus(task_db)
        if not selected:
            return False

        # 用户 GPU 权限检查（非管理员走白名单）
        user = task_db.user
        if user.role != Role.ADMIN:
            visible = set(user.get_visible_gpus_list())
            if not all(gid in visible for gid in selected):
                return False

        if not resources_available(selected, task_db.gpu_memory_limit):
            return False

        # 状态更新为 RUNNING
        task_db.status = TaskStatus.RUNNING
        task_db.started_at = dt.datetime.now(dt.UTC)
        session.commit()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected)

        proc = subprocess.Popen(
            task_db.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=task_db.working_dir or os.getcwd(),
            env=env,
            text=True,
        )
        out, err = proc.communicate()

        task_db.finished_at = dt.datetime.now(dt.UTC)
        task_db.stdout_log = out
        task_db.stderr_log = err
        task_db.status = TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
        session.commit()
    except Exception:
        session.rollback()
        raise
    else:
        return True
    finally:
        session.close()


def scheduler_loop(poll_interval: float, session_local: sessionmaker[Session]) -> None:
    """
    调度器主循环：查找 PENDING 任务，按 priority（降序）和 created_at（升序）调度。
    """
    while True:
        session: Session = session_local()
        try:
            tasks = (
                session.query(Task)
                .filter(Task.status == TaskStatus.PENDING)
                .order_by(Task.priority.desc(), Task.created_at.asc())
                .all()
            )
            for task in tasks:
                if allocate_and_run_task(task, session_local):
                    continue
                # TODO 分配失败：可能资源不够或权限不足，留待下轮
        finally:
            session.close()
        time.sleep(poll_interval)
