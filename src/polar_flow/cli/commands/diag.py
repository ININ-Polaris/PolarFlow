from typing import TYPE_CHECKING

import typer
from rich import print

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_kv

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

app = typer.Typer(help="Cluster diagnostics")


@app.command("ping")
def ping(ctx: typer.Context) -> None:
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/ping/")
    print_kv("PING", data, cfg.logging.dict_style)


@app.command("diag")
def diag(ctx: typer.Context) -> None:
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/diag/")
    print_kv("diag", data, cfg.logging.dict_style)
