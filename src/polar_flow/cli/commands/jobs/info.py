from typing import TYPE_CHECKING, Annotated

import typer

from polar_flow._vendor.slurm_client.models.slurm_v0043_get_job_flags import SlurmV0043GetJobFlags
from polar_flow._vendor.slurm_client.models.slurm_v0043_get_jobs_flags import SlurmV0043GetJobsFlags
from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import (
    PrintProgress,
    print_client_we,
    print_json_ex,
)

from ..ann import job_info_ann  # noqa: TID252

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

from .jobs import job_app


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    update_time: Annotated[
        None | str,
        typer.Option(
            ...,
            "--utime",
            help="最近更新的查询作业（UNIX时间戳）",
        ),
    ] = None,
    flags: Annotated[
        None | SlurmV0043GetJobsFlags,
        typer.Option(..., "--flags", help="查询标志位"),
    ] = None,
    vvv: Annotated[bool, typer.Option(..., "--vvv", help="列出完整信息（会非常多）")] = False,
) -> None:
    """列出作业"""
    progress_msg = "[bold cyan]请稍候，正在获取...[/]"
    if vvv:
        progress_msg = "[bold cyan]请稍候，正在获取详细信息...[/]"

    with PrintProgress(progress_msg):
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get_jobs(update_time=update_time, flags=flags)
        jobs = data.jobs
        errors = data.errors
        warnings = data.warnings
        print_client_we(warnings=warnings, errors=errors)

    filtered_jobs = []
    for job in jobs:
        if not vvv:
            filtered_jobs.append(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "user_name": job.user_name,
                    "group_name": job.group_name,
                    "job_state": job.job_state,
                    "state_reason": job.state_reason,
                },
            )
        else:
            filtered_jobs.append(job.to_dict())

    title = "作业列表"
    print_json_ex(
        title,
        data={"jobs": filtered_jobs},
        key_priority=["jobs"],
        expand=True,
        show_raw=debug,
        annotations=job_info_ann,
        show_side_notes_for_tables=True,
        notes_panel_title="注释",
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
        dict_notes_panel_title="相关信息",
        table_max_keys=8,
    )


@job_app.command("show")
def job_show(
    ctx: typer.Context,
    job_id: int = typer.Argument(..., help="作业 ID"),
    update_time: Annotated[
        None | str,
        typer.Option(
            ...,
            "--utime",
            help="最近更新的查询作业（UNIX时间戳）",
        ),
    ] = None,
    flags: Annotated[
        None | SlurmV0043GetJobFlags,
        typer.Option(..., "--flags", help="查询标志位"),
    ] = None,
    vvv: Annotated[bool, typer.Option(..., "--vvv", help="列出完整信息（会非常多）")] = False,
) -> None:
    """查看单个作业详情"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get_job(job_id=str(job_id), update_time=update_time, flags=flags)
        jobs = data.jobs
        errors = data.errors
        warnings = data.warnings
        print_client_we(warnings=warnings, errors=errors)

    filtered_jobs = jobs
    max_cols = 8 if vvv else 4
    filtered_jobs = []
    for job in jobs:
        if not vvv:
            filtered_jobs.append(
                {
                    "command": job.command,
                    "current_working_directory": job.current_working_directory,
                    "flags": job.flags,
                    "job_id": job.job_id,
                    "hold": job.hold,
                    "group_id": job.group_id,
                    "group_name": job.group_name,
                    "job_state": job.job_state,
                    "state_reason": job.state_reason,
                    "tres_req_str": job.tres_req_str,
                    "name": job.name,
                    "user_name": job.user_name,
                    "account": job.account,
                    "partition": job.partition,
                    "qos": job.qos,
                },
            )
        else:
            filtered_jobs.append(job.to_dict())

    print_json_ex(
        "作业详情",
        data={"jobs": filtered_jobs},
        key_priority=["jobs"],
        expand=True,
        show_raw=debug,
        annotations=job_info_ann,
        show_side_notes_for_tables=True,
        notes_panel_title="注释",
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
        dict_notes_panel_title="相关信息",
        table_max_keys=max_cols,
    )
