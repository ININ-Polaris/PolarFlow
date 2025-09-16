from typing import TYPE_CHECKING

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_kv

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

acct_app = typer.Typer(help="账户/用户/QOS/TRES 等（需要 slurmdbd 端点）")


@acct_app.command("accounts")
def acct_accounts(ctx: typer.Context) -> None:
    """列出账户（GET /slurmdb/v0.0.43/accounts/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug, prefix="/slurmdb")  # 若客户端支持前缀
    data = c.get("/accounts/")
    print_kv("账户列表", data, cfg.logging.dict_style)


@acct_app.command("users")
def acct_users(ctx: typer.Context) -> None:
    """列出用户（GET /slurmdb/v0.0.43/users/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug, prefix="/slurmdb")
    data = c.get("/users/")
    print_kv("用户列表", data, cfg.logging.dict_style)


@acct_app.command("qos")
def acct_qos(ctx: typer.Context) -> None:
    """列出 QOS（GET /slurmdb/v0.0.43/qos/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug, prefix="/slurmdb")
    data = c.get("/qos/")
    print_kv("QOS 列表", data, cfg.logging.dict_style)


@acct_app.command("tres")
def acct_tres(ctx: typer.Context) -> None:
    """列出 TRES（GET /slurmdb/v0.0.43/tres/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug, prefix="/slurmdb")
    data = c.get("/tres/")
    print_kv("TRES 列表", data, cfg.logging.dict_style)
