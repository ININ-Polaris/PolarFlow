from typing import TYPE_CHECKING, Annotated

import typer

from polar_flow._vendor.slurm_client.models.slurm_v0043_delete_job_flags import (
    SlurmV0043DeleteJobFlags,
)
from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import (
    PrintProgress,
    print_client_we,
    print_json_ex,
)

from .ann import submit_ann

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

from .jobs import job_app


@job_app.command("cancel")
def job_cancel(
    ctx: typer.Context,
    job_id: int = typer.Argument(..., help="作业 ID"),
    signal: str | None = typer.Option(
        None,
        "--signal",
        help="发送信号而非直接取消，例如 TERM,KILL",
    ),
    flags: Annotated[
        None | SlurmV0043DeleteJobFlags,
        typer.Option(..., "--flags", help="过滤标志位"),
    ] = None,
) -> None:
    """取消/信号作业"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.delete_job(job_id=str(job_id), signal=signal, flags=flags)
        errors = data.errors
        warnings = data.warnings
        print_client_we(warnings=warnings, errors=errors)
        if len(data.status) == 1:
            status = data.status[0].to_dict()
        elif len(data.status) == 0:
            status = {
                "job_id": job_id,
                "status": "Success",
            }
        else:
            status = [s.to_dict() for s in data.status]

    print_json_ex(
        "操作结果",
        data={"result": status},
        key_priority=["result"],
        expand=True,
        show_raw=debug,
        annotations=submit_ann,
        show_side_notes_for_tables=True,
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
    )
