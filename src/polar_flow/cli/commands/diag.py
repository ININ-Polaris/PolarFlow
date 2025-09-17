from enum import Enum
from typing import TYPE_CHECKING, Annotated

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import PrintProgress, print_json_ex

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

cluster_app = typer.Typer(help="集群诊断与控制")


@cluster_app.command("ping")
def ping(ctx: typer.Context) -> None:
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get("/ping/")

    from .ann import ping_ann  # noqa: PLC0415

    print_json_ex(
        "基本信息",
        data={"pings": data["pings"]},
        key_priority=["pings"],
        expand=True,
        show_raw=debug,
        annotations=ping_ann,
        show_side_notes_for_tables=True,
        notes_panel_title="注释",
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
        dict_notes_panel_title="相关信息",
    )


class ShowMode(str, Enum):
    all = "all"
    meta = "meta"
    stat = "stat"


@cluster_app.command("diag")
def diag(
    ctx: typer.Context,
    show: Annotated[
        ShowMode,
        typer.Option(
            ...,
            "--show",
            "-s",
            help="格式化（控制显示哪些数据）",
        ),
    ] = ShowMode.meta,
) -> None:
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]

        c = SlurmClient(cfg, token, debug=debug)
        data = c.get("/diag/")

    if show in (ShowMode.all, ShowMode.meta):
        from .ann import meta_ann  # noqa: PLC0415

        print_json_ex(
            "基本信息",
            data=data["meta"],
            key_priority=["plugin", "client", "command", "slurm"],
            expand=True,
            show_raw=debug,
            annotations=meta_ann,
            show_side_notes_for_tables=True,
            notes_panel_title="注释",
            show_side_notes_for_dicts=True,
            dict_notes_min_hits=2,
            dict_notes_max_depth=3,
            dict_notes_panel_title="相关信息",
        )
    if show in (ShowMode.all, ShowMode.stat):
        from .ann import stats_ann  # noqa: PLC0415

        print_json_ex(
            "统计数据",
            data={"statistics": data["statistics"]},
            key_priority=["statistics"],
            expand=True,
            show_raw=debug,
            annotations=stats_ann,
            show_side_notes_for_tables=True,
            notes_panel_title="注释",
            show_side_notes_for_dicts=True,
            dict_notes_min_hits=2,
            dict_notes_max_depth=3,
            dict_notes_panel_title="相关信息",
        )
