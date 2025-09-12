# server/scheduler.py
from __future__ import annotations

import datetime as dt
import os
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from polar_flow.server.gpu_monitor import get_all_gpu_info
from polar_flow.server.models import Role, Task, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

SessionFactory = Callable[[], Session]

# 进程内 GPU 占用表与互斥锁（防止同一 worker 内重复分配）
ALLOCATED: set[int] = set()
ALLOC_LOCK = threading.Lock()


@contextmanager
def reserve_gpus(gids: list[int]):  # noqa: ANN201
    with ALLOC_LOCK:
        if any(g in ALLOCATED for g in gids):
            yield False
            return
        for g in gids:
            ALLOCATED.add(g)
    try:
        yield True
    finally:
        with ALLOC_LOCK:
            for g in gids:
                ALLOCATED.discard(g)


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


def _spawn_and_track(task_db: Task, selected: list[int], session_local: SessionFactory) -> None:
    """在后台线程内等待进程结束并写回结果。"""
    env = os.environ.copy()
    # CUDA 设备与额外可观测变量（见修复 #4）
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected)
    env["POLAR_ALLOCATED_GPU_IDS"] = ",".join(str(x) for x in selected)
    env["NVIDIA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]

    # 使用新会话组，便于取消时整组终止
    proc = subprocess.Popen(
        task_db.command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=task_db.working_dir or os.getcwd(),
        env=env,
        text=True,
        start_new_session=True,
    )

    # 把 pid 写回数据库，便于 cancel 杀进程（见修复 #2）
    session: Session = session_local()
    try:
        t = session.get(Task, task_db.id)
        if t:
            t.pid = proc.pid
            session.commit()
    finally:
        session.close()

    out, err = proc.communicate()

    # 结果回写：日志落盘 + DB 仅保存摘要（见修复 #12）
    session = session_local()
    try:
        t = session.get(Task, task_db.id)
        if not t:
            return
        t.finished_at = dt.datetime.now(dt.UTC)
        from .utils_logging import save_task_logs  # 新增工具，见修复 #12

        stdout_path, stderr_path, out_snip, err_snip = save_task_logs(t, out, err)
        t.stdout_path = stdout_path
        t.stderr_path = stderr_path
        t.stdout_log = out_snip
        t.stderr_log = err_snip
        t.status = TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
        session.commit()
    finally:
        session.close()


def allocate_and_run_task(
    task: Task, session_local: SessionFactory, async_run: bool = False
) -> bool:
    session: Session = session_local()
    try:
        # 在当前 session 中把 task 捞出来（顺便把 user 一并 eager load，避免再次懒加载）
        task_db = session.execute(
            select(Task).options(joinedload(Task.user)).where(Task.id == task.id),
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

        # 进程内资源占位，且做原子状态 CAS（见修复 #10）
        with reserve_gpus(selected) as ok:
            if not ok:
                return False
            # 原子更新：仅当当前仍为 PENDING 才置 RUNNING
            rows = (
                session.query(Task)
                .filter(Task.id == task_db.id, Task.status == TaskStatus.PENDING)
                .update(
                    {
                        Task.status: TaskStatus.RUNNING,
                        Task.started_at: dt.datetime.now(dt.UTC),
                    },
                    synchronize_session=False,
                )
            )
            session.commit()
            if rows == 0:
                # 状态已被他处更改（可能 CANCELLED），放弃
                return False

            if async_run:
                # 后台线程处理执行与回写
                threading.Thread(
                    target=_spawn_and_track, args=(task_db, selected, session_local), daemon=True
                ).start()
            else:
                # —— 同步分支（保持向后兼容，供单测/调用方期待立即得到 SUCCESS/FAILED）——
                env = os.environ.copy()
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected)
                env["POLAR_ALLOCATED_GPU_IDS"] = ",".join(str(x) for x in selected)
                env["NVIDIA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]

                proc = subprocess.Popen(
                    task_db.command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=task_db.working_dir or os.getcwd(),
                    env=env,
                    text=True,
                    start_new_session=True,
                )
                # 写回 pid，供取消使用
                task_row = session.get(Task, task_db.id)
                if task_row:
                    task_row.pid = proc.pid
                    session.commit()

                out, err = proc.communicate()

                # 结果回写：落盘 + DB 摘要
                task_row = session.get(Task, task_db.id)
                if task_row:
                    task_row.finished_at = dt.datetime.now(dt.UTC)
                    from .utils_logging import save_task_logs

                    stdout_path, stderr_path, out_snip, err_snip = save_task_logs(
                        task_row, out, err
                    )
                    task_row.stdout_path = stdout_path
                    task_row.stderr_path = stderr_path
                    task_row.stdout_log = out_snip
                    task_row.stderr_log = err_snip
                    task_row.status = (
                        TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
                    )
                    session.commit()
            return True
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
                if allocate_and_run_task(task, session_local, True):
                    continue
                # TODO 分配失败：可能资源不够或权限不足，留待下轮
        finally:
            session.close()
        time.sleep(poll_interval)
