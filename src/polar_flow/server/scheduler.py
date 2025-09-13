# server/scheduler.py
from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from polar_flow.server.gpu_monitor import get_all_gpu_info
from polar_flow.server.models import Role, Task, TaskStatus
from polar_flow.server.utils_logging import (
    format_argv,
    redact_env,
    save_task_logs,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

SessionFactory = Callable[[], Session]

# 进程内 GPU 占用表与互斥锁（防止同一 worker 内重复分配）
ALLOCATED: set[int] = set()
ALLOC_LOCK = threading.Lock()


logger = logging.getLogger(__name__)


@contextmanager
def reserve_gpus(gids: list[int]) -> Generator[bool]:
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
    if not requested:
        return True
    infos = get_all_gpu_info()
    logger.debug(
        "resources_available: requested=%s, gpu_mem_limit=%s, nvml_count=%s",
        requested,
        gpu_memory_limit,
        len(infos),
    )
    free_map: dict[int, int] = {g["id"]: g["memory_free"] for g in infos}  # bytes

    for gid in requested:
        free_bytes = free_map.get(gid)
        if free_bytes is None:
            return False
        if gpu_memory_limit is not None:
            required_bytes = gpu_memory_limit * 1024 * 1024  # MB -> bytes
            if free_bytes < required_bytes:
                logger.debug(
                    "resources_available: gid=%s insufficient free=%s < required=%s",
                    gid,
                    free_bytes,
                    required_bytes,
                )
                return False
    return True


def build_command_and_env_for_task(
    task_db: Task,
    selected: list[int],
) -> tuple[list[str], dict[str, str]]:
    """
    返回 (argv, env)：
      - Host:   ["bash","-lc", <script>]
      - Docker: ["docker","run",...]
    """
    # 1) 合成 host 侧 env
    env = os.environ.copy()
    # 叠加任务环境变量（支持 $HOME 展开）
    if task_db.env:
        for k, v in task_db.env.items():
            env[k] = os.path.expandvars(v)

    # GPU/CPU-only 环境变量（Host 侧）
    if selected:
        # Host 进程里，如需本地执行，按主机索引过滤
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected)
        env["POLAR_ALLOCATED_GPU_IDS"] = env["CUDA_VISIBLE_DEVICES"]
        # 不再在 Host 侧主动设置 NVIDIA_VISIBLE_DEVICES，避免与 nvidia runtime 冲突
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
    else:
        # CPU-only：清理 GPU 相关变量
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
        env.pop("CUDA_DEVICE_ORDER", None)
        env["POLAR_ALLOCATED_GPU_IDS"] = ""

    # 2) 拼脚本
    script = task_db.command

    # 3) Host 模式（不走 Docker）
    if not task_db.docker_image:
        return (["bash", "-lc", script], env)

    # 4) Docker 模式
    workdir = task_db.working_dir or os.getcwd()
    img = task_db.docker_image
    args = list(task_db.docker_args or [])

    cmd = ["docker", "run", "--rm"]

    # 4.1 GPU：只用 --gpus 选择主机 GPU（让 nvidia-container-runtime 处理映射与注入）
    if selected:
        # 这里传入主机索引；容器内会被重编号为 0..N-1
        device_arg = "device=" + ",".join(map(str, selected))
        cmd += ["--gpus", device_arg]
    # else: CPU-only 不传 --gpus

    # 4.2 工作目录
    cmd += ["-v", f"{workdir}:/work", "-w", "/work"]

    # 4.3 透传到容器的 env（与 host 侧区分开来，避免把主机索引带进容器）
    container_env = {}

    # 任务自定义 env（优先）
    if task_db.env:
        for k, v in task_db.env.items():
            container_env[k] = os.path.expandvars(v)

    # 常用变量
    if "HOME" in env:
        container_env["HOME"] = env["HOME"]

    # 始终透传排序策略
    container_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # 仅作为记录/诊断：保留主机侧分配信息
    container_env["POLAR_ALLOCATED_GPU_IDS"] = ",".join(map(str, selected)) if selected else ""

    # 关键点：如果容器里需要 CUDA_VISIBLE_DEVICES，按容器重编号设置为 0..N-1
    # 这既满足部分框架对该变量的依赖，又不会与 --gpus 的实际映射冲突。
    if selected:
        remapped = ",".join(str(i) for i in range(len(selected)))  # e.g. "0" or "0,1"
        container_env["CUDA_VISIBLE_DEVICES"] = remapped
    else:
        # CPU-only：不设置该变量
        pass

    # 不要覆盖 NVIDIA_VISIBLE_DEVICES；由 runtime 注入
    # Docker 内部会重映射序号

    # 4.4 将 container_env 写入 docker -e
    for k in sorted(container_env.keys()):
        cmd += ["-e", f"{k}={container_env[k]}"]

    # 4.5 追加额外 docker args（如 --ipc=host 等）
    cmd += args

    # 4.6 镜像 + 入口
    cmd += [img, "bash", "-lc", script]

    # 返回 docker 命令（容器内 env 的注入在 cmd 中），以及 host 侧 env（供调用方需要时使用）
    return (cmd, env)


def _select_gpus(task: Task) -> list[int]:
    kind = task.requested_gpus.strip().upper()
    if kind in ("CPU", "NONE"):
        logger.debug("select_gpus: CPU-only for task id=%s", task.id)
        return []
    if task.requested_gpus.startswith("AUTO:"):
        num = int(task.requested_gpus.split(":", 1)[1])
        infos = get_all_gpu_info()

        # 注意：NVML 是字节，这里做单位换算
        limit_bytes = None
        if task.gpu_memory_limit is not None:
            limit_bytes = task.gpu_memory_limit * 1024 * 1024

        # 默认候选：满足显存限制的 GPU
        candidates = [g for g in infos if (limit_bytes is None or g["memory_free"] >= limit_bytes)]

        # 非管理员：再按用户可见集过滤，避免选到越权的 GPU（比如 4）
        if task.user and task.user.role != Role.ADMIN:
            visible = set(task.user.get_visible_gpus_list() or [])
            candidates = [g for g in candidates if g["id"] in visible]
        if len(candidates) < num:
            logger.debug("select_gpus: not enough candidates need=%s got=%s", num, len(candidates))
            return []
        selected = [
            g["id"] for g in sorted(candidates, key=lambda x: x["memory_free"], reverse=True)[:num]
        ]
        logger.debug("select_gpus: AUTO selected=%s (limit_bytes=%s)", selected, limit_bytes)
    else:
        selected = [int(x) for x in task.requested_gpus.split(",") if x.strip() != ""]
    return selected


def _spawn_and_track(task_db: Task, selected: list[int], session_local: SessionFactory) -> None:
    """在后台线程内等待进程结束并写回结果。"""
    # 使用新会话组，便于取消时整组终止；env 统一由构建函数产出
    argv, env = build_command_and_env_for_task(task_db, selected)

    mode = "docker" if task_db.docker_image else "host"
    img = task_db.docker_image or ""
    cwd = task_db.working_dir or os.getcwd()

    pass_env_keys = set((task_db.env or {}).keys()) | {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "POLAR_ALLOCATED_GPU_IDS",
        "HOME",
    }
    logger.info(
        "exec(prepare): user=%s mode=%s docker_image=%s gpus=%s cwd=%s argv=%s env_excerpt=%s",
        getattr(getattr(task_db, "user", None), "username", None),
        mode,
        img,
        selected,
        cwd,
        format_argv(argv),
        redact_env(env, pass_env_keys),
    )
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=task_db.working_dir or os.getcwd(),  # 对 docker 来说不影响；对 host 有效
        env=env,
        text=True,
        start_new_session=True,
    )

    # 把 pid 写回数据库，便于 cancel 杀进程
    session: Session = session_local()
    try:
        t = session.get(Task, task_db.id)
        if t:
            t.pid = proc.pid
            session.commit()
            logger.info(
                "task[%s] started pid=%s selected_gpus=%s cwd=%s docker=%s",
                t.id,
                proc.pid,
                selected,
                task_db.working_dir,
                task_db.docker_image,
            )
    finally:
        session.close()

    out, err = proc.communicate()

    # 结果回写：日志落盘 + DB 仅保存摘要
    session = session_local()
    try:
        t = session.get(Task, task_db.id)
        if not t:
            return
        t.finished_at = dt.datetime.now(dt.UTC)

        stdout_path, stderr_path, out_snip, err_snip = save_task_logs(t, out, err)
        t.stdout_path = stdout_path
        t.stderr_path = stderr_path
        t.stdout_log = out_snip
        t.stderr_log = err_snip
        t.status = TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
        session.commit()
        logger.info("task[%s] finished rc=%s status=%s", t.id, proc.returncode, t.status)
    finally:
        session.close()


def allocate_and_run_task(
    task: Task,
    session_local: SessionFactory,
    async_run: bool = False,
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
        cpu_only = task_db.requested_gpus.strip().upper() in ("CPU", "NONE")
        if not selected and not cpu_only:
            logger.debug(
                "allocate: selection failed task_id=%s req=%s",
                task_db.id,
                task_db.requested_gpus,
            )
            return False

        # 用户 GPU 权限检查（非管理员走白名单）
        user = task_db.user
        if user.role != Role.ADMIN:
            visible = set(user.get_visible_gpus_list())
            if selected and not all(gid in visible for gid in selected):
                logger.debug(
                    "allocate: user %s lacks gpu visibility selected=%s visible=%s",
                    user.username,
                    selected,
                    sorted(visible),
                )
                return False

        if not resources_available(selected, task_db.gpu_memory_limit):
            return False

        # 进程内资源占位，且做原子状态 CAS
        with reserve_gpus(selected) as ok:
            if not ok:
                logger.debug("allocate: reserve_gpus busy selected=%s", selected)
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
                logger.debug("allocate: CAS lost task_id=%s", task_db.id)
                return False

            if async_run:
                # 后台线程处理执行与回写
                threading.Thread(
                    target=_spawn_and_track,
                    args=(task_db, selected, session_local),
                    daemon=True,
                ).start()
                logger.debug(
                    "allocate: spawned async runner task_id=%s selected=%s",
                    task_db.id,
                    selected,
                )
            else:
                # —— 同步分支（保持向后兼容，供单测/调用方期待立即得到 SUCCESS/FAILED）——
                argv, env = build_command_and_env_for_task(task_db, selected)
                mode = "docker" if task_db.docker_image else "host"
                img = task_db.docker_image or ""
                cwd = task_db.working_dir or os.getcwd()
                pass_env_keys = set((task_db.env or {}).keys()) | {
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "POLAR_ALLOCATED_GPU_IDS",
                    "HOME",
                }
                logger.info(
                    "exec(prepare): user=%s mode=%s docker_image=%s gpus=%s cwd=%s argv=%s env_excerpt=%s",
                    getattr(getattr(task_db, "user", None), "username", None),
                    mode,
                    img,
                    selected,
                    cwd,
                    format_argv(argv),
                    redact_env(env, pass_env_keys),
                )
                proc = subprocess.Popen(
                    argv,
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
                    logger.info(
                        "task[%s] started (sync) pid=%s selected_gpus=%s cwd=%s",
                        task_row.id,
                        proc.pid,
                        selected,
                        task_db.working_dir,
                    )

                out, err = proc.communicate()

                # 结果回写：落盘 + DB 摘要
                task_row = session.get(Task, task_db.id)
                if task_row:
                    task_row.finished_at = dt.datetime.now(dt.UTC)

                    stdout_path, stderr_path, out_snip, err_snip = save_task_logs(
                        task_row,
                        out,
                        err,
                    )
                    task_row.stdout_path = stdout_path
                    task_row.stderr_path = stderr_path
                    task_row.stdout_log = out_snip
                    task_row.stderr_log = err_snip
                    task_row.status = (
                        TaskStatus.SUCCESS if proc.returncode == 0 else TaskStatus.FAILED
                    )
                    session.commit()
                    logger.info(
                        "task[%s] finished (sync) rc=%s status=%s",
                        task_row.id,
                        proc.returncode,
                        task_row.status,
                    )
            return True
    except Exception:
        session.rollback()
        logger.exception(
            "allocate: exception task_id=%s",
            getattr(task, "id", None),
        )
        raise
    else:
        return True
    finally:
        session.close()


def preview_task_command_and_env(task_db: Task) -> tuple[str, dict[str, str]]:
    """
    干跑预览：不执行任务，只返回日志中会打印的两项：
      - format_argv(argv)
      - redact_env(env, pass_env_keys)

    返回:
        (argv_formatted, env_excerpt)
    """
    # 计算将要使用的 GPU（保持与实际路径一致，但不做权限/资源占用）
    selected = _select_gpus(task_db)

    # 构建命令与环境（与真实执行一致）
    argv, env = build_command_and_env_for_task(task_db, selected)

    # 与日志相同的 env 过滤键集合
    pass_env_keys = set((task_db.env or {}).keys()) | {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "POLAR_ALLOCATED_GPU_IDS",
        "HOME",
    }

    # 返回日志里两项
    return (
        format_argv(argv),
        redact_env(env, pass_env_keys),
    )


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
            logger.debug("scheduler: pending=%s", len(tasks))
            for task in tasks:
                ok = allocate_and_run_task(task, session_local, True)
                logger.debug("scheduler: try task_id=%s ok=%s", task.id, ok)
                if ok:
                    continue
                # TODO 分配失败：可能资源不够或权限不足，留待下轮
        finally:
            session.close()
        time.sleep(poll_interval)
