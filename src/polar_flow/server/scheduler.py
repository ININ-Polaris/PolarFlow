# server/scheduler.py
from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from .gpu_monitor import get_all_gpu_info
from .models import Role, Task, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

SessionFactory = Callable[[], Session]


def resources_available(requested: list[int], gpu_memory_limit: int | None) -> bool:
    infos = get_all_gpu_info()
    free_map: dict[int, int] = {g["id"]: g["memory_free"] for g in infos}
    for gid in requested:
        free = free_map.get(gid)
        if free is None:
            return False
        if gpu_memory_limit is not None and free < gpu_memory_limit:
            return False
    return True


def _select_gpus(task: Task) -> list[int]:
    if task.requested_gpus.startswith("AUTO:"):
        num = int(task.requested_gpus.split(":", 1)[1])
        infos = get_all_gpu_info()
        candidates = [
            g
            for g in infos
            if (task.gpu_memory_limit is None or g["memory_free"] >= task.gpu_memory_limit)
        ]
        if len(candidates) < num:
            return []
        selected = [
            g["id"] for g in sorted(candidates, key=lambda x: x["memory_free"], reverse=True)[:num]
        ]
    else:
        selected = [int(x) for x in task.requested_gpus.split(",") if x.strip() != ""]
    return selected


def allocate_and_run_task(task: Task, session_local: sessionmaker[Session]) -> bool:
    """
    尝试为 task 分配 GPU 并启动。如果成功返回 True，否则 False。
    """
    session: Session = session_local()
    try:
        selected = _select_gpus(task)
        if not selected:
            return False

        # 用户 GPU 权限检查（非管理员走白名单）
        user = task.user
        if user.role != Role.ADMIN:
            visible = set(user.get_visible_gpus_list())
            if not all(gid in visible for gid in selected):
                return False

        if not resources_available(selected, task.gpu_memory_limit):
            return False

        # 状态更新为 RUNNING
        task.status = TaskStatus.RUNNING
        task.started_at = dt.datetime.now(dt.UTC)
        session.add(task)
        session.commit()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in selected
        )  # 官方推荐变量名，控制可见与重编号
        # 参考 NVIDIA 文档：CUDA_VISIBLE_DEVICES 会决定可见设备及其重编号顺序
        # https://docs.nvidia.com/deploy/topics/topic_5_2_1.html

        proc = subprocess.Popen(
            task.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=task.working_dir or os.getcwd(),
            env=env,
            text=True,
        )
        out, err = proc.communicate()

        task.finished_at = dt.datetime.now(dt.UTC)
        task.stdout_log = out
        task.stderr_log = err
        task.status = TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
        session.add(task)
        session.commit()
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
