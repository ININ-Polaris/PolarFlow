# create_admin.py
import argparse
from pathlib import Path

from polar_flow.server.config import Config
from polar_flow.server.db import create_session_factory
from polar_flow.server.models import Base, Role, User


def main() -> None:
    parser = argparse.ArgumentParser(description="创建或覆盖管理员账户")
    parser.add_argument("--username", "-u", default="admin", help="管理员用户名")
    parser.add_argument("--password", "-p", default="admin123", help="管理员密码（请务必修改！）")
    parser.add_argument("--force", action="store_true", help="若存在则强制覆盖")
    args = parser.parse_args()

    config_path = "data/config.toml"
    cfg = Config.load(Path(config_path) if config_path else Path("config.toml"))
    session_local, engine = create_session_factory(cfg.server.database_url)
    Base.metadata.create_all(engine)
    s = session_local()
    try:
        u = s.query(User).filter(User.username == args.username).first()
        if u:
            if not args.force:
                print(f"管理员 '{args.username}' 已经存在，使用 --force 参数可覆盖。")
                return
            # 覆盖账号
            u.role = Role.ADMIN
            u.priority = 100
            u.visible_gpus = []
            u.set_password(args.password)
            s.commit()
            print(f"管理员 '{args.username}' 已更新密码。")
        else:
            u = User(
                username=args.username,
                role=Role.ADMIN,
                priority=100,
                visible_gpus=[],
            )
            u.set_password(args.password)
            s.add(u)
            s.commit()
            print("管理员已创建：", args.username, args.password)
    finally:
        s.close()
