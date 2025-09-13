# app.py
# app.py
from __future__ import annotations

import importlib
from pathlib import Path

from flask import Flask, Response, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

importlib.import_module("polar_flow.server.models")

from polar_flow.server.auth import auth_bp, login_manager, set_session_factory  # noqa: E402
from polar_flow.server.config import Config  # noqa: E402
from polar_flow.server.models import Base  # noqa: E402
from polar_flow.server.schemas import UserRead  # noqa: E402


# -------- App Factory --------
def create_app(config_path: str) -> Flask:
    app = Flask(__name__)

    # 1) 加载配置
    cfg = Config.load(Path(config_path) if config_path else Path("config.toml"))
    print(cfg)
    app.config["SECRET_KEY"] = cfg.server.secret_key

    # 2) 初始化数据库（Engine / Session 工厂）
    engine = create_engine(cfg.server.database_url, future=True)
    session_local: sessionmaker[Session] = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )

    # 如需自动建表（开发阶段可用，生产建议迁移脚本）
    Base.metadata.create_all(bind=engine)

    # 3) 注入会话工厂给认证模块，注册 Flask-Login
    set_session_factory(session_local)
    login_manager.init_app(app)

    # 4) 注册蓝图
    from polar_flow.server.routes import (  # noqa: PLC0415
        api_bp,
        set_session_factory as routes_set_session_factory,
    )
    routes_set_session_factory(session_local)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # 5) 演示路由：健康检查
    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"status": "ok"})

    # 6) 演示路由：查看当前用户信息（需要登录）
    from flask_login import current_user, login_required  # noqa: PLC0415

    @app.get("/me")
    @login_required
    def me() -> Response:
        return jsonify(UserRead.model_validate(current_user).model_dump())

    return app


def main() -> None:
    app = create_app("data/config.toml")
    # 生产环境请使用 WSGI/ASGI 服务器；这里用于本地开发
    app.run(host="0.0.0.0", port=5000, debug=True)



# config.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import toml
from pydantic import BaseModel, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    secret_key: str = Field(...)
    database_url: str = Field(...)
    redis_url: str = Field(...)
    scheduler_poll_interval: int = Field(
        ..., gt=0, description="轮询间隔（秒），必须大于 0",
    )

    @field_validator("scheduler_poll_interval", mode="after")
    @classmethod
    def check_positive_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("scheduler_poll_interval 必须大于 0")
        return v


class DefaultsConfig(BaseModel):
    user_priority: int = Field(
        default=100,
        ge=0,
        description="默认普通用户提交任务可用的最大优先级(>= 0)",
    )

    @field_validator("user_priority", mode="after")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("user_priority 必须大于等于 0")
        return v


class Config(BaseModel):
    server: ServerConfig
    defaults: DefaultsConfig

    @classmethod
    def load(cls, config_path: Path) -> Config:
        data = {}
        if config_path.exists():
            try:
                data = toml.load(config_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"无法加载配置文件 {config_path}: {e}; 将使用默认配置")
        else:
            logger.warning(f"未找到 {config_path}; 将使用默认配置")

        basedir = config_path.parent

        server_data = data.get("server", {})

        secret_key = server_data.get(
            "secret_key",
            os.environ.get("SECRET_KEY", "you-will-never-guess"),
        )

        database_url = server_data.get("database_url") or os.environ.get("DATABASE_URL")
        if not database_url:
            database_url = f"sqlite:///{(basedir / 'app.db').as_posix()}"

        redis_url = server_data.get("redis_url") or os.environ.get(
            "REDIS_URL", "redis://localhost:6379/0",
        )

        # 统一解析 scheduler_poll_interval
        spi_raw = server_data.get("scheduler_poll_interval")
        if spi_raw is None:
            spi_raw = os.environ.get("SCHEDULER_POLL_INTERVAL")
        try:
            spi = int(spi_raw) if spi_raw is not None else 5
        except ValueError:
            logger.warning(
                f"scheduler_poll_interval 无法解析为整数: {spi_raw}, 使用默认 5",
            )
            spi = 5
        try:
            spi = int(spi_raw) if spi_raw is not None else 5
        except ValueError:
            logger.warning(
                f"scheduler_poll_interval 无法解析为整数: {spi_raw}, 使用默认 5",
            )
            spi = 5

        # 统一解析 user_priority
        up_raw = (data.get("defaults", {}) or {}).get("user_priority")
        try:
            up = int(up_raw) if up_raw is not None else 100
        except ValueError:
            logger.warning(f"user_priority 无法解析为整数: {up_raw}, 使用默认 100")
            up = 100

        try:
            return cls(
                server=ServerConfig(
                    secret_key=secret_key,
                    database_url=database_url,
                    redis_url=redis_url,
                    scheduler_poll_interval=spi,
                ),
                defaults=DefaultsConfig(user_priority=up),
            )
        except ValidationError:
            logger.exception("配置文件解析错误")
            raise


# server/gpu_monitor.py

from __future__ import annotations

import logging
import time
from typing import TypedDict

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)

logger = logging.getLogger(__name__)


class GPUInfo(TypedDict):
    id: int
    memory_total: int
    memory_free: int
    memory_used: int
    util_gpu: int
    util_mem: int


def get_all_gpu_info() -> list[GPUInfo]:
    """
    返回所有 GPU 的状态列表
    """
    try:
        nvmlInit()
    except NVMLError:
        logger.exception("初始化 GPU 失败")
        return []

    gpu_count = nvmlDeviceGetCount()
    infos: list[GPUInfo] = []
    for i in range(gpu_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        info = GPUInfo(
            id=i,
            memory_total=int(mem.total),
            memory_free=int(mem.free),
            memory_used=int(mem.used),
            util_gpu=int(util.gpu),
            util_mem=int(util.memory),
        )
        infos.append(info)
    return infos


def monitor_loop(poll_interval: float = 5.0) -> None:
    """
    后台线程 /进程执行 GPU 信息监控，
    定期（默认每 poll_interval 秒）采集并可供调度器 /网页 UI 查询
    """
    while True:
        infos = get_all_gpu_info()
        # TODO 把 infos 存到全局缓存 /共享状态里
        print("GPU infos:", infos)
        time.sleep(poll_interval)


# server/models.py
from __future__ import annotations

import datetime as dt
from enum import Enum

from flask_login import UserMixin
from sqlalchemy import JSON, DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Role(Enum):
    USER = "user"
    ADMIN = "admin"


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class User(Base, UserMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[Role] = mapped_column(SAEnum(Role), default=Role.USER, nullable=False)
    visible_gpus: Mapped[list[int]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,  # 注意：用可调用对象，避免共享同一个列表
        nullable=False,
    )
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    # 注意：前向引用用字符串，避免静态类型检查报错
    tasks: Mapped[list[Task]] = relationship(
        "Task",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def set_password(self, raw: str) -> None:
        from werkzeug.security import generate_password_hash  # noqa: PLC0415

        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        from werkzeug.security import check_password_hash  # noqa: PLC0415

        return check_password_hash(self.password_hash, raw)

    def get_visible_gpus_list(self) -> list[int]:
        return self.visible_gpus


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    user: Mapped[User] = relationship("User", back_populates="tasks")

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    command: Mapped[str] = mapped_column(String(512), nullable=False)
    requested_gpus: Mapped[str] = mapped_column(String(64), nullable=False)  # "0,1" 或 "AUTO:2"
    gpu_memory_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)  # MB
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    working_dir: Mapped[str] = mapped_column(String(256), nullable=False)

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus),
        default=TaskStatus.PENDING,
        nullable=False,
    )

    # 使用时区感知时间（UTC），并设置 timezone=True
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC),
        nullable=False,
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    stdout_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr_log: Mapped[str | None] = mapped_column(Text, nullable=True)


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


# server/schemas.py
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from polar_flow.server.models import Role, TaskStatus  # noqa: TC001

if TYPE_CHECKING:
    import datetime as dt


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6)

    model_config = ConfigDict(extra="forbid")


class UserRead(BaseModel):
    id: int
    username: str
    role: Role
    visible_gpus: list[int] = Field(default_factory=list)
    priority: int

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    command: str = Field(..., min_length=1)
    requested_gpus: str = Field(..., min_length=1)
    working_dir: str = Field(..., min_length=1, max_length=256)
    gpu_memory_limit: int | None = Field(default=None, ge=0)
    priority: int = Field(default=100, ge=0)

    model_config = ConfigDict(extra="forbid")


class TaskRead(BaseModel):
    id: int
    user_id: int
    name: str
    command: str
    requested_gpus: str
    working_dir: str
    gpu_memory_limit: int | None
    priority: int
    status: TaskStatus
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    stdout_log: str | None
    stderr_log: str | None

    model_config = ConfigDict(from_attributes=True)


# server/worker.py
from __future__ import annotations

from pathlib import Path

from polar_flow.server.config import Config
from polar_flow.server.db import create_session_factory
from polar_flow.server.models import Base
from polar_flow.server.scheduler import scheduler_loop


def run_worker(config_path: str | None = None) -> None:
    cfg = Config.load(Path(config_path)) if config_path else Config.load(Path("config.toml"))
    poll_interval = cfg.server.scheduler_poll_interval
    session_local, engine = create_session_factory(cfg.server.database_url)
    Base.metadata.create_all(engine)  # ensure tables exist
    scheduler_loop(poll_interval=poll_interval, session_local=session_local)

# server/db.py
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(database_url: str) -> tuple[sessionmaker[Session], Engine]:
    """Helper: 创建 SQLAlchemy session 工厂与 engine。
    在 worker 与 app 两边均可重用。
    """
    engine = create_engine(database_url, future=True)
    session_local: sessionmaker[Session] = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    return session_local, engine

# server/auth.py
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from flask import Blueprint, Response, jsonify, request
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from polar_flow.server.models import Role, User
from polar_flow.server.schemas import UserRead

if TYPE_CHECKING:
    from collections.abc import Callable

    from flask.typing import ResponseReturnValue
    from sqlalchemy.orm import Session, sessionmaker

# ---- Flask-Login 基础对象 ----
auth_bp = Blueprint("auth", __name__)
login_manager = LoginManager()


@login_manager.unauthorized_handler
def _unauthorized():  # noqa: ANN202
    return jsonify({"error": "login required"}), 401


# ---- 会话工厂注入 ----
_session_factory: sessionmaker[Session] | None = None


def set_session_factory(session_factory: sessionmaker[Session]) -> None:
    """在应用初始化阶段调用，一次性注入会话工厂。"""
    global _session_factory  # noqa: PLW0603
    _session_factory = session_factory


def _get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("Session factory is not initialized. Call set_session_factory() first.")
    return _session_factory()


def get_user_by_username(username: str) -> User | None:
    session = _get_session()
    try:
        return session.query(User).filter(User.username == username).first()
    finally:
        session.close()


@auth_bp.route("/auth/login", methods=["POST"])
def login() -> tuple[Response, int]:
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400

    user = get_user_by_username(username)
    if user is None or not user.check_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)

    print(user.visible_gpus)

    return jsonify(
        {"message": "logged in", "user": UserRead.model_validate(user).model_dump()},
    ), 200


@auth_bp.route("/auth/logout", methods=["POST"])
@login_required
def logout() -> tuple[Response, int]:
    logout_user()
    return jsonify({"message": "logged out"}), 200


def admin_required[**P](func: Callable[P, ResponseReturnValue]) -> Callable[P, ResponseReturnValue]:
    """
    管理员权限校验装饰器：
    - 使用 ParamSpec 保留被装饰函数的参数签名（*args/**kwargs 的类型信息）
    - 返回类型采用 Flask 的 ResponseReturnValue（str | bytes | Response | (Response, status) ...）
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResponseReturnValue:
        if not current_user.is_authenticated:
            return jsonify({"error": "login required"}), 401
        if getattr(current_user, "role", None) != Role.ADMIN:
            return jsonify({"error": "admin required"}), 403
        return func(*args, **kwargs)

    return wrapper


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    session = _get_session()
    try:
        return session.get(User, uid)
    finally:
        session.close()


# server/routes.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Blueprint, Response, jsonify, request
from flask_login import current_user, login_required

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
    if payload.requested_gpus.startswith("AUTO:"):
        try:
            n = int(payload.requested_gpus.split(":", 1)[1])
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus AUTO:<n> 格式错误"}), 400
        if n <= 0:
            return jsonify({"error": "AUTO 台数必须 > 0"}), 400
    else:
        try:
            _ = [int(x) for x in payload.requested_gpus.split(",") if x.strip()]
        except Exception:  # noqa: BLE001
            return jsonify({"error": "requested_gpus 需为 '0,1' 或 'AUTO:n'"}), 400

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
        )
        sess.add(task)
        sess.commit()
        sess.refresh(task)
        return jsonify(TaskRead.model_validate(task).model_dump()), 201
    finally:
        sess.close()


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
        return jsonify([TaskRead.model_validate(t).model_dump() for t in items])
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
        return jsonify(TaskRead.model_validate(t).model_dump()), 200
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
        # 这里只是把状态改为 CANCELLED；运行中进程的终止由更完善的执行器实现。
        t.status = TaskStatus.CANCELLED
        t.finished_at = dt.datetime.now(dt.UTC)
        sess.commit()
        return jsonify({"message": "cancelled"}), 200
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
        return jsonify(UserRead.model_validate(u).model_dump()), 201
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
        return jsonify(UserRead.model_validate(u).model_dump()), 200
    finally:
        sess.close()


@api_bp.get("/admin/users")
@admin_required
def list_users() -> Response:
    sess = _get_session()
    try:
        items = sess.query(User).order_by(User.id.asc()).all()
        return jsonify([UserRead.model_validate(u).model_dump() for u in items])
    finally:
        sess.close()


# cli/entry.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import colorama
import toml
import typer
import requests

app = typer.Typer(add_completion=False, help="BIT ININ 课题组自用 服务器 GPU 资源分配器")

DEFAULT_BASE_URL = os.environ.get("POLAR_BASE_URL", "http://127.0.0.1:5000")
STATE_DIR = Path(os.environ.get("POLAR_STATE_DIR", "~/.polar_flow")).expanduser()
COOKIE_FILE = STATE_DIR / "cookies.txt"


class Client:
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if COOKIE_FILE.exists():
            try:
                self.session.cookies.update(
                    requests.utils.cookiejar_from_dict(json.loads(COOKIE_FILE.read_text()))
                )
            except Exception:
                pass

    def _save_cookies(self) -> None:
        COOKIE_FILE.write_text(json.dumps(requests.utils.dict_from_cookiejar(self.session.cookies)))

    # ---- Auth ----
    def login(self, username: str, password: str) -> dict:
        r = self.session.post(
            f"{self.base_url}/auth/login", json={"username": username, "password": password}
        )
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # 尝试把服务端返回的错误正文打印出来
            try:
                err = r.json().get("error")
            except Exception:
                err = r.text
            raise SystemExit(f"{colorama.Fore.BLUE}[登录失败]: {colorama.Fore.RED}{err} ({e}){colorama.Style.RESET_ALL}")

    def logout(self) -> dict:
        r = self.session.post(f"{self.base_url}/auth/logout")
        r.raise_for_status()
        self._save_cookies()
        return r.json()

    # ---- Tasks ----
    def create_task(self, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks", json=payload)
        r.raise_for_status()
        return r.json()

    def list_tasks(self, status: Optional[str] = None) -> list[dict]:
        params = {"status": status} if status else None
        r = self.session.get(f"{self.base_url}/api/tasks", params=params)
        r.raise_for_status()
        return r.json()

    def get_task(self, task_id: int) -> dict:
        r = self.session.get(f"{self.base_url}/api/tasks/{task_id}")
        r.raise_for_status()
        return r.json()

    def cancel_task(self, task_id: int) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks/{task_id}/cancel")
        r.raise_for_status()
        return r.json()

    def list_gpus(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/api/gpus")
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # 尝试把服务端返回的错误正文打印出来
            try:
                err = r.json().get("error")
            except Exception:
                err = r.text
            raise SystemExit(f"{colorama.Fore.BLUE}[查询失败]: {colorama.Fore.RED}{err} ({e}){colorama.Style.RESET_ALL}")
        return r.json()


# ---------------- CLI commands ----------------
@app.command()
def login(
    username: str = typer.Option(..., "--username", "-u"),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
):
    """登录并保存会话 Cookie。"""
    c = Client(base_url)
    res = c.login(username, password)
    typer.echo(f"Logged in as {res['user']['username']}")


@app.command()
def logout(base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url")):
    """注销当前会话。"""
    c = Client(base_url)
    try:
        res = c.logout()
        typer.echo(res.get("message", "logged out"))
    except requests.HTTPError as e:
        typer.echo(f"Logout failed: {e}")


@app.command("gpus")
def gpus_cmd(base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url")):
    """查看 GPU 状态。"""
    c = Client(base_url)
    infos = c.list_gpus()
    for g in infos:
        typer.echo(json.dumps(g, ensure_ascii=False))


@app.command("submit")
def submit_cmd(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True, help="TOML 任务配置文件"
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
):
    """从 TOML 提交任务。"""
    data = toml.load(config)
    # 支持：task.name, task.command, task.requested_gpus, task.working_dir, task.gpu_memory_limit, task.priority
    t = data.get("task", {})
    payload = {
        "name": t.get("name"),
        "command": t.get("command"),
        "requested_gpus": t.get("requested_gpus", "AUTO:1"),
        "working_dir": t.get("working_dir", str(Path.cwd())),
        "gpu_memory_limit": t.get("gpu_memory_limit"),
        "priority": t.get("priority", 100),
    }
    c = Client(base_url)
    res = c.create_task(payload)
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2))


@app.command("ls")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", help="过滤任务状态(PENDING/RUNNING/SUCCESS/FAILED/CANCELLED)"
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
):
    """列出我的任务。"""
    c = Client(base_url)
    items = c.list_tasks(status=status)
    for it in items:
        typer.echo(
            f"#{it['id']} [{it['status']}] {it['name']}  prio={it['priority']}  created={it['created_at']}"
        )


@app.command("logs")
def logs_cmd(
    task_id: int = typer.Argument(...), base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url")
):
    """查看任务日志（stdout/stderr）。"""
    c = Client(base_url)
    t = c.get_task(task_id)
    typer.echo(f"== stdout ==\n{t.get('stdout_log') or ''}\n")
    typer.echo(f"== stderr ==\n{t.get('stderr_log') or ''}\n")


@app.command("cancel")
def cancel_cmd(
    task_id: int = typer.Argument(...), base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url")
):
    c = Client(base_url)
    res = c.cancel_task(task_id)
    typer.echo(res.get("message", "ok"))


def main() -> None:
    app()
