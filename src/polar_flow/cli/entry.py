import os
from pathlib import Path
import sys

import typer

from polar_flow.cli.config import Token, is_jwt_expired, load_config, load_token, save_token
from polar_flow.cli.printers import print_error, print_warning

from .auth import app as auth_app, login
from .commands import diag, jobs, res

DEBUG = False

app = typer.Typer(
    add_completion=False,
    help="BIT ININ GPU 资源管理工具 [beta 1.0.0]",
)


def _is_auth_login_invocation(argv: list[str]) -> bool:
    """是否是 `auth login` 调用（允许前面有全局选项），以及帮助场景"""
    # 如果全局是在看帮助，直接放行
    if any(x in argv for x in ("-h", "--help")):
        return True

    # 找到第一个不以 '-' 开头的 token 作为一级命令
    i = 0
    while i < len(argv) and argv[i].startswith("-"):
        # 跳过选项的值：诸如 --config PATH（当 PATH 不以 '-' 开头时）
        if argv[i] in ("-c", "--config") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            i += 2
        else:
            i += 1

    if i >= len(argv):
        return False  # 没有子命令

    # 一级子命令
    if argv[i] != "auth":
        return False

    # 可能是 `auth --help` 之类
    rest = argv[i + 1 :]
    if not rest or any(x in rest for x in ("-h", "--help")):
        return True  # 看 help 时放行

    # 二级子命令需为 login
    return rest[0] == "login"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML 配置文件路径; default: env POLAR_CONFIG_PATH or ~/.config/polarflow/config.toml",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Debug mode",
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    app.pretty_exceptions_show_locals = debug

    # 1) 先判断是否是允许跳过鉴权的命令
    argv = sys.argv[1:]
    is_auth_login = _is_auth_login_invocation(argv)

    # 2) 加载配置（保持你原逻辑）
    try:
        if config:
            cfg = load_config(Path(config))
        else:
            cfg = load_config(
                Path(os.environ.get("POLAR_CONFIG_PATH", "~/.config/polarflow/config.toml")),
            )
    except FileNotFoundError:
        with open(Path("~/.config/polarflow/config.toml").expanduser(), "w", encoding="utf-8") as f:
            f.write("[pam-server]\n")
            f.write('host = "pam.server"\n')
            f.write("port = 6602\n\n")
            f.write("[slurm-server]\n")
            f.write('host = "slurm.server"\n')
            f.write("port = 6601\n\n")
            f.write("[logging]\n")
            f.write('dict_style = "table"\n')
        print_warning(
            "未找到配置文件，默认配置文件已写入 ~/.config/polarflow/config.toml",
            "配置缺失",
        )
        raise typer.Exit(1) from None

    ctx.obj["cfg"] = cfg

    # 3) 鉴权
    if not is_auth_login:
        token = load_token()
        if token is None:
            print_error("请先登录: auth login", "权限认证失败")
            raise typer.Exit(-1)

        if is_jwt_expired(token):
            print_error("请重新登录: auth login", "权限认证过期")
            raise typer.Exit(-1)

        ctx.obj["token"] = token.jwt


app.add_typer(auth_app, name="auth")
app.add_typer(diag.cluster_app, name="diag")
app.add_typer(jobs.job_app, name="jobs")
app.add_typer(res.res_app, name="res")
# app.add_typer(nodes.node_app, name="nodes")
# app.add_typer(partitions.partition_app, name="partitions")
# app.add_typer(reservation.reservation_app, name="reservation")
# app.add_typer(acct.acct_app, name="accounting")


def entry() -> None:
    app()
