from typing import TYPE_CHECKING, Any

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_kv, print_kv_grouped

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

cluster_app = typer.Typer(help="集群诊断与控制")


@cluster_app.command("ping")
def ping(ctx: typer.Context) -> None:
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/ping/")
    payload: dict[str, Any] = {}
    payload["节点"] = {
        "节点名": data["meta"]["slurm"]["cluster"],
        "客户端": data["meta"]["client"]["source"],
    }
    pings = data["pings"]
    for ping in pings:
        payload[ping["hostname"]] = {"状态": ping["pinged"], "延迟": f"{ping['latency']} 毫秒"}

    print_kv_grouped("PING", payload, cfg.logging.dict_style, group_order=["节点"])


@cluster_app.command("diag")
def diag(ctx: typer.Context) -> None:
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/diag/")
    print_kv("diag", data, cfg.logging.dict_style)
