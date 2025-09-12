from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, ParamSpec

from flask import Blueprint, Response, jsonify, request
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from .models import Role, User
from .schemas import UserRead

if TYPE_CHECKING:
    from collections.abc import Callable

    from flask.typing import ResponseReturnValue
    from sqlalchemy.orm import Session, sessionmaker

P = ParamSpec("P")

# ---- Flask-Login 基础对象 ----
auth_bp = Blueprint("auth", __name__)
login_manager = LoginManager()

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
        # 也可使用 check_password_hash(user.password_hash, password)
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)
    return jsonify(
        {"message": "logged in", "user": UserRead.model_validate(user).model_dump()}
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
    """Flask-Login 要求用 user_id 恢复用户会话。"""
    session = _get_session()
    try:
        # 使用 get(primary_key) 按 ID 加载
        return session.get(User, int(user_id))
    finally:
        session.close()
