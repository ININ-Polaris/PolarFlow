# server/routes.py
from __future__ import annotations

import datetime as dt
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flask import Blueprint, Response, jsonify, request
from flask_login import current_user, login_required

from polar_flow.server.scheduler import preview_task_command_and_env

from .auth import admin_required
from .models import Role, Task, TaskStatus, User
from .schemas import TaskCreate, TaskRead, UserCreate, UserRead

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

# 通过 app.py 在应用初始化阶段注入 session factory
_session_factory: sessionmaker[Session] | None = None


def set_session_factory(session_factory: sessionmaker[Session]) -> None:
    global _session_factory  # noqa: PLW0603
    _session_factory = session_factory


def _get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("routes: Session factory is not initialized")
    return _session_factory()


api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------- GPU 可见性与健康 ----------
@api_bp.get("/gpus")
@login_required
def list_gpus() -> Response:
    from .gpu_monitor import get_all_gpu_info  # 延迟导入避免 NVML 成本  # noqa: PLC0415

    infos = get_all_gpu_info()
    return jsonify(infos)


# ---------- 任务 CRUD（当前用户域） ----------
@api_bp.post("/tasks")
@login_required
def create_task() -> tuple[Response, int]:
    data = request.json or {}
    try:
        payload = TaskCreate.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"invalid payload: {e}"}), 400

    # 基础校验
    kind = payload.requested_gpus.strip().upper()
    if kind in ("CPU", "NONE"):
        pass
    elif payload.requested_gpus.startswith("AUTO:"):
        try:
            n = int(payload.requested_gpus.split(":", 1)[1])
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus AUTO:<n> 格式错误"}), 400
        if n <= 0:
            return jsonify({"error": "AUTO 台数必须 > 0"}), 400
    else:
        try:
            gids = [int(x) for x in payload.requested_gpus.split(",") if x.strip()]
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus 需为 '0,1' 或 'AUTO:n'"}), 400
        if current_user.role != Role.ADMIN:
            visible = set(current_user.get_visible_gpus_list() or [])
            if not all(g in visible for g in gids):
                return jsonify({"error": "所选 GPU 超出可见范围"}), 403

    # 非管理员不可越权设定优先级
    priority = payload.priority
    if current_user.role != Role.ADMIN and priority > current_user.priority:
        priority = current_user.priority

    # working_dir 必须存在
    if not Path(payload.working_dir).exists():
        return jsonify({"error": f"working_dir 不存在: {payload.working_dir}"}), 400

    sess = _get_session()
    try:
        task = Task(
            user_id=current_user.id,
            name=payload.name,
            command=payload.command,
            requested_gpus=payload.requested_gpus,
            gpu_memory_limit=payload.gpu_memory_limit,
            priority=priority,
            working_dir=str(Path(payload.working_dir).resolve()),
            status=TaskStatus.PENDING,
            docker_image=payload.docker_image,
            docker_args=payload.docker_args or None,
            env=payload.env or None,
        )
        sess.add(task)
        sess.commit()
        sess.refresh(task)
        return jsonify(TaskRead.model_validate(task).model_dump(mode="json")), 201
    finally:
        sess.close()


@api_bp.post("/tasks_check")
@login_required
def check_task() -> tuple[Response, int]:
    data = request.json or {}
    try:
        payload = TaskCreate.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"invalid payload: {e}"}), 400

    # 基础校验
    kind = payload.requested_gpus.strip().upper()
    if kind in ("CPU", "NONE"):
        pass
    elif payload.requested_gpus.startswith("AUTO:"):
        try:
            n = int(payload.requested_gpus.split(":", 1)[1])
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus AUTO:<n> 格式错误"}), 400
        if n <= 0:
            return jsonify({"error": "AUTO 台数必须 > 0"}), 400
    else:
        try:
            gids = [int(x) for x in payload.requested_gpus.split(",") if x.strip()]
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus 需为 '0,1' 或 'AUTO:n'"}), 400
        if current_user.role != Role.ADMIN:
            visible = set(current_user.get_visible_gpus_list() or [])
            if not all(g in visible for g in gids):
                return jsonify({"error": "所选 GPU 超出可见范围"}), 403

    # 非管理员不可越权设定优先级
    priority = payload.priority
    if current_user.role != Role.ADMIN and priority > current_user.priority:
        priority = current_user.priority

    # working_dir 必须存在
    if not Path(payload.working_dir).exists():
        return jsonify({"error": f"working_dir 不存在: {payload.working_dir}"}), 400

    task = Task(
        user_id=current_user.id,
        name=payload.name,
        command=payload.command,
        requested_gpus=payload.requested_gpus,
        gpu_memory_limit=payload.gpu_memory_limit,
        priority=priority,
        working_dir=str(Path(payload.working_dir).resolve()),
        status=TaskStatus.PENDING,
        docker_image=payload.docker_image,
        docker_args=payload.docker_args or None,
        env=payload.env or None,
    )
    task.user = cast("User", current_user._get_current_object())

    argv_fmt, _ = preview_task_command_and_env(task)

    return jsonify({"cmd": argv_fmt}), 200


@api_bp.get("/tasks")
@login_required
def list_tasks() -> tuple[Response, int] | Response:
    """列出当前用户的任务；管理员可查看全部并按用户过滤。"""
    user_id = request.args.get("user_id", type=int)
    status = request.args.get("status")

    sess = _get_session()
    try:
        q = sess.query(Task)
        if current_user.role != Role.ADMIN:
            q = q.filter(Task.user_id == current_user.id)
        elif user_id:
            q = q.filter(Task.user_id == user_id)
        if status:
            try:
                st = TaskStatus(status)
                q = q.filter(Task.status == st)
            except Exception:  # noqa: BLE001
                return jsonify({"error": "status 无效"}), 400
        q = q.order_by(Task.created_at.desc())
        items = q.all()
        return jsonify([TaskRead.model_validate(t).model_dump(mode="json") for t in items])
    finally:
        sess.close()


@api_bp.get("/tasks/<int:task_id>")
@login_required
def get_task(task_id: int) -> tuple[Response, int]:
    sess = _get_session()
    try:
        t = sess.get(Task, task_id)
        if not t:
            return jsonify({"error": "not found"}), 404
        if current_user.role != Role.ADMIN and t.user_id != current_user.id:
            return jsonify({"error": "forbidden"}), 403
        return jsonify(TaskRead.model_validate(t).model_dump(mode="json")), 200
    finally:
        sess.close()


@api_bp.post("/tasks/<int:task_id>/cancel")
@login_required
def cancel_task(task_id: int) -> tuple[Response, int]:
    sess = _get_session()
    try:
        t = sess.get(Task, task_id)
        if not t:
            return jsonify({"error": "not found"}), 404
        if current_user.role != Role.ADMIN and t.user_id != current_user.id:
            return jsonify({"error": "forbidden"}), 403
        if t.status in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return jsonify({"message": f"task already {t.status.value}"}), 200
        # 尝试终止真实进程（仅 RUNNING 有意义）
        if t.status == TaskStatus.RUNNING and t.pid:
            try:
                # 优雅终止整个进程组
                if os.name != "nt":
                    os.killpg(t.pid, signal.SIGTERM)
                else:
                    os.kill(t.pid, signal.SIGTERM)  # Windows 上由 Python 映射
            except Exception:  # noqa: BLE001
                pass
            # 等待最多 10*0.2=2s
            for _ in range(10):
                try:
                    os.kill(t.pid, 0)
                    time.sleep(0.2)
                except Exception:  # noqa: BLE001
                    break
            else:
                # 仍存活则强杀
                try:
                    if os.name != "nt":
                        os.killpg(t.pid, signal.SIGKILL)
                    else:
                        os.kill(t.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        t.status = TaskStatus.CANCELLED
        t.finished_at = dt.datetime.now(dt.UTC)
        sess.commit()
        return jsonify({"message": "cancelled ok"}), 200
    finally:
        sess.close()


# ---------- 用户管理（仅管理员） ----------
@api_bp.post("/admin/users")
@admin_required
def create_user() -> tuple[Response, int]:
    data = request.json or {}
    try:
        payload = UserCreate.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"invalid payload: {e}"}), 400

    sess = _get_session()
    try:
        if sess.query(User).filter(User.username == payload.username).first():
            return jsonify({"error": "username exists"}), 409
        u = User(
            username=payload.username,
            role=Role.USER,
            priority=100,
            visible_gpus=[],
        )
        u.set_password(payload.password)
        sess.add(u)
        sess.commit()
        sess.refresh(u)
        return jsonify(UserRead.model_validate(u).model_dump(mode="json")), 201
    finally:
        sess.close()


@api_bp.get("/admin/users/<int:user_id>")
@admin_required
def get_user_admin(user_id: int) -> tuple[Response, int]:
    sess = _get_session()
    try:
        u = sess.get(User, user_id)
        if not u:
            return jsonify({"error": "not found"}), 404
        return jsonify(UserRead.model_validate(u).model_dump(mode="json")), 200
    finally:
        sess.close()


@api_bp.patch("/admin/users/<int:user_id>")
@admin_required
def patch_user(user_id: int) -> tuple[Response, int]:
    data = request.json or {}
    sess = _get_session()
    try:
        u = sess.get(User, user_id)
        if not u:
            return jsonify({"error": "not found"}), 404
        # 允许修改：role, priority, visible_gpus, password
        if "role" in data:
            try:
                u.role = Role(data["role"])  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                return jsonify({"error": "role must be 'user'|'admin'"}), 400
        if "priority" in data:
            try:
                p = int(data["priority"])
                if p < 0:
                    raise ValueError  # noqa: TRY301
                u.priority = p
            except Exception:  # noqa: BLE001
                return jsonify({"error": "priority must be >= 0"}), 400
        if "visible_gpus" in data:
            v = data["visible_gpus"]
            if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
                return jsonify({"error": "visible_gpus must be int list"}), 400
            u.visible_gpus = v
        if "password" in data:
            u.set_password(str(data["password"]))
        sess.commit()
        return jsonify(UserRead.model_validate(u).model_dump(mode="json")), 200
    finally:
        sess.close()


@api_bp.get("/admin/users")
@admin_required
def list_users() -> Response:
    sess = _get_session()
    try:
        items = sess.query(User).order_by(User.id.asc()).all()
        return jsonify([UserRead.model_validate(u).model_dump(mode="json") for u in items])
    finally:
        sess.close()
