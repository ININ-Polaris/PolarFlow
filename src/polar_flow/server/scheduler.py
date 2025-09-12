# server/scheduler.py

from __future__ import annotations

import datetime as dt
import os
import subprocess
import time
from typing import TYPE_CHECKING

from .gpu_monitor import get_all_gpu_info
from .models import Role, Task, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


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


def allocate_and_run_task(task: Task) -> bool:
    """
    尝试为 task 分配 GPU 并启动。如果成功返回 True，否则 False。
    """
    session: Session = SessionLocal()
    try:
        # 确定 requested gpus
        if task.requested_gpus.startswith("AUTO:"):
            num = int(task.requested_gpus.split(":", 1)[1])
            infos = get_all_gpu_info()
            candidates = [
                g
                for g in infos
                if (task.gpu_memory_limit is None or g["memory_free"] >= task.gpu_memory_limit)
            ]
            if len(candidates) < num:
                return False
            # 按 空闲显存 排序选最大
            selected = [
                g["id"]
                for g in sorted(candidates, key=lambda x: x["memory_free"], reverse=True)[:num]
            ]
        else:
            selected = [int(x) for x in task.requested_gpus.split(",")]

        # 检查用户是否有权限使用这些 GPU
        user = task.user
        if user.role != Role.ADMIN:  # 假设你的 Role 是类内部定义或导出的
            visible = user.get_visible_gpus_list()
            for gid in selected:
                if gid not in visible:
                    return False

        if not resources_available(selected, task.gpu_memory_limit):
            return False

        # 更新状态，启动任务
        task.status = TaskStatus.RUNNING
        task.started_at = dt.datetime.now(tz=dt.UTC)
        session.add(task)
        session.commit()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected)

        proc = subprocess.Popen(
            task.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=task.working_dir if hasattr(task, "working_dir") else os.getcwd(),
            env=env,
            text=True,
        )
        out, err = proc.communicate()

        task.finished_at = datetime.utcnow()
        task.stdout_log = out
        task.stderr_log = err
        task.status = TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
        session.add(task)
        session.commit()
        return True
    finally:
        session.close()


def scheduler_loop(poll_interval: float) -> None:
    """
    调度器主循环：查找 PENDING 任务，按 priority 排序，试图分配资源并启动它们。
    """
    while True:
        session: Session = SessionLocal()
        try:
            tasks = (
                session.query(Task)
                .filter(Task.status == TaskStatus.PENDING)
                .order_by(Task.priority.desc(), Task.created_at)
                .all()
            )
            for task in tasks:
                allocated = allocate_and_run_task(task)
                if allocated:
                    # 如果启动成功就继续其他 task
                    continue
                # 失败的话 skip，下一轮再试
            # 睡眠
        finally:
            session.close()
        time.sleep(poll_interval)
