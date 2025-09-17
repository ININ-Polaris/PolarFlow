import os
from pathlib import Path

import typer

from .auth import app as auth_app
from .commands import acct, diag, jobs, nodes, partitions, reservation

DEBUG = False

app = typer.Typer(
    add_completion=True,
    help="BIT ININ 自用 SLURM 包装",
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

    if config:
        cfg = load_config(Path(config))
    else:
        cfg = load_config(
            Path(os.environ.get("POLAR_CONFIG_PATH", "~/.config/polarflow/config.toml")),
        )

    token = load_token()

    ctx.obj["cfg"] = cfg
    ctx.obj["token"] = token


app.add_typer(auth_app, name="auth")
app.add_typer(diag.cluster_app, name="diag")
app.add_typer(jobs.job_app, name="jobs")
app.add_typer(nodes.node_app, name="nodes")
app.add_typer(partitions.partition_app, name="partitions")
app.add_typer(reservation.reservation_app, name="reservation")
app.add_typer(acct.acct_app, name="accounting")


def entry() -> None:
    app()
