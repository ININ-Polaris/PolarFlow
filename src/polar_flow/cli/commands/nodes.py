from typing import TYPE_CHECKING, Any

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_kv, print_warning

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

node_app = typer.Typer(help="计算节点查看/维护")


@node_app.command("list")
def node_list(ctx: typer.Context) -> None:
    """列出所有节点（/nodes/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/nodes/")
    print_kv("节点列表", data, cfg.logging.dict_style)


@node_app.command("show")
def node_show(ctx: typer.Context, node_name: str = typer.Argument(..., help="节点名")) -> None:
    """查看单个节点（/node/{node_name}）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get(f"/node/{node_name}")
    print_kv(f"节点 {node_name}", data, cfg.logging.dict_style)


@node_app.command("update")
def node_update(
    ctx: typer.Context,
    node_name: str = typer.Argument(..., help="节点名"),
    state: str | None = typer.Option(None, "--state", help="目标状态，如 DRAIN,DOWN,RESUME"),
    reason: str | None = typer.Option(None, "--reason", help="状态变更原因"),
) -> None:
    """更新节点属性/状态（POST /node/{node_name} 或批量 /nodes/）"""
    if state is None and reason is None:
        print_warning("未提供任何更新字段，跳过")
        return
    body: dict[str, Any] = {}
    if state:
        body["node_state"] = state
    if reason:
        body["reason"] = reason
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    resp = c.post_json(f"/node/{node_name}", body=body)
    print_kv("更新结果", resp, cfg.logging.dict_style)
