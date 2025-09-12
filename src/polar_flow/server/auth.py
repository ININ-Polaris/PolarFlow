from __future__ import annotations

from flask import Blueprint, Response, jsonify, request
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash

from .models import Role, User
from .schemas import UserRead


def get_user_by_username(username: str, session_local) -> User | None:
    session = session_local()
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
    if user is None or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)
    # 或者如果你用 token 认证，就生成 token
    # 下面假设你返回一个 simple token 或者 session cookie
    return jsonify({"message": "logged in", "user": UserRead.model_validate(user).dict()}), 200


@auth_bp.route("/auth/logout", methods=["POST"])
@login_required
def logout() -> tuple[Response, int]:
    logout_user()
    return jsonify({"message": "logged out"}), 200


def admin_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "login required"}), 401
        if current_user.role != Role.ADMIN:
            return jsonify({"error": "admin required"}), 403
        return func(*args, **kwargs)

    return wrapper


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    session = SessionLocal()
    try:
        return session.query(User).get(int(user_id))
    finally:
        session.close()
