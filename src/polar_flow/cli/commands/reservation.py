from typing import TYPE_CHECKING

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_error, print_kv

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

reservation_app = typer.Typer(help="预留操作")


@reservation_app.command("list")
def reservation_list(ctx: typer.Context) -> None:
    """列出预留（/reservations/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/reservations/")
    print_kv("预留列表", data, cfg.logging.dict_style)


@reservation_app.command("show")
def reservation_show(ctx: typer.Context, name: str = typer.Argument(..., help="预留名")) -> None:
    """查看预留（/reservation/{name}）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get(f"/reservation/{name}")
    print_kv(f"预留 {name}", data, cfg.logging.dict_style)


@reservation_app.command("apply")
def reservation_apply(
    ctx: typer.Context,
    spec_file: str = typer.Argument(..., help="JSON 规格文件，符合 slurmrestd 预留模式"),
) -> None:
    """创建/更新预留（POST /reservation 或 /reservations/）"""
    try:
        import json  # noqa: PLC0415

        with open(spec_file, encoding="utf-8") as file:
            spec = json.load(file)
    except Exception as e:  # noqa: BLE001
        print_error(f"读取规格失败：{e}")
        raise typer.Exit(code=1) from None
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    resp = c.post_json("/reservation", body=spec)
    print_kv("预留结果", resp, cfg.logging.dict_style)


@reservation_app.command("delete")
def reservation_delete(ctx: typer.Context, name: str = typer.Argument(..., help="预留名")) -> None:
    """删除预留（DELETE /reservation/{name}）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    resp = c.delete(f"/reservation/{name}")
    print_kv("删除结果", resp, cfg.logging.dict_style)
