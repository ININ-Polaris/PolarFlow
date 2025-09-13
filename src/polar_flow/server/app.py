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
        return jsonify(UserRead.model_validate(current_user).model_dump(mode="json"))

    return app


def main() -> None:
    app = create_app("data/config.toml")
    # 生产环境请使用 WSGI/ASGI 服务器；这里用于本地开发
    app.run(host="0.0.0.0", port=5000, debug=True)
