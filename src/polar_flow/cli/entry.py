import os
from pathlib import Path

import typer

from polar_flow.cli.config import Token, is_jwt_expired, save_token
from polar_flow.cli.printers import print_error, print_warning

from .auth import app as auth_app
from .commands import diag, jobs, res

DEBUG = False

app = typer.Typer(
    add_completion=False,
    help="BIT ININ GPU 资源管理工具 [beta 0.0.6]",
)


@app.callback()
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
    from .config import load_config, load_token  # noqa: PLC0415

    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    app.pretty_exceptions_show_locals = debug

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
            f.write('host = "10.1.74.164"\n')
            f.write("port = 6602\n")
            f.write("\n")
            f.write("[slurm-server]\n")
            f.write('host = "10.1.74.164"\n')
            f.write("port = 6601\n")
            f.write("\n")
            f.write("[logging]\n")
            f.write('dict_style = "table"\n')
        print_warning(
            "未找到配置文件，默认配置文件已写入 ~/.config/polarflow/config.toml",
            "配置缺失",
        )
        raise typer.Exit(1) from None

    token = load_token()

    if token is None:
        save_token(Token("", 0))
        print_error("请先登录: auth login", "权限认证失败")
        raise typer.Exit(-1)

    if is_jwt_expired(token):
        print_error("请重新登录: auth login", "权限认证过期")
        raise typer.Exit(-1)

    ctx.obj["cfg"] = cfg
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
