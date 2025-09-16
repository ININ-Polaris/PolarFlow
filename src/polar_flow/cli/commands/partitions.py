from typing import TYPE_CHECKING

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_kv

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

partition_app = typer.Typer(help="分区查看")


@partition_app.command("list")
def partition_list(ctx: typer.Context) -> None:
    """列出分区（/partitions/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/partitions/")
    print_kv("分区列表", data, cfg.logging.dict_style)


@partition_app.command("show")
def partition_show(
    ctx: typer.Context,
    partition_name: str = typer.Argument(..., help="分区名"),
) -> None:
    """查看单个分区（/partition/{partition_name}）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get(f"/partition/{partition_name}")
    print_kv(f"分区 {partition_name}", data, cfg.logging.dict_style)
