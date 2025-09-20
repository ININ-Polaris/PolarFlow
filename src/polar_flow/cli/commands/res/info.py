from typing import TYPE_CHECKING

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import (
    PrintProgress,
    print_client_we,
    print_json_ex,
)

from .ann import res_ann
from .res import res_app

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig


@res_app.command("list")
def list_tres(ctx: typer.Context) -> None:
    """检测可用资源状态"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get_tres()
        errors = data.errors
        warnings = data.warnings
        tres = data.tres

    print_client_we(warnings=warnings, errors=errors)

    print_json_ex(
        "资源信息",
        data={"tres": [t.to_dict() for t in tres]},
        key_priority=["tres"],
        expand=True,
        show_raw=debug,
        annotations=res_ann,
        show_side_notes_for_tables=True,
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
    )
