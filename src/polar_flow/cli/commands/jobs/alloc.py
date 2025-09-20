from typing import TYPE_CHECKING

import typer

from polar_flow._vendor.slurm_client.models.v0043_job_alloc_req import V0043JobAllocReq
from polar_flow._vendor.slurm_client.models.v0043_job_desc_msg import V0043JobDescMsg
from polar_flow._vendor.slurm_client.models.v0043_job_desc_msg_kill_warning_flags_item import (
    V0043JobDescMsgKillWarningFlagsItem,
)
from polar_flow._vendor.slurm_client.models.v0043_job_desc_msg_open_mode_item import (
    V0043JobDescMsgOpenModeItem,
)
from polar_flow._vendor.slurm_client.models.v0043_job_desc_msg_shared_item import (
    V0043JobDescMsgSharedItem,
)
from polar_flow.cli.client import SlurmClient
from polar_flow.cli.commands.jobs.read_toml_script import job_desc_from_toml
from polar_flow.cli.commands.jobs.utils import parse_noval_ui32, parse_noval_ui64, parse_time_type
from polar_flow.cli.printers import (
    PrintProgress,
    print_client_we,
    print_debug,
    print_json_ex,
)

from .ann import submit_ann
from .jobs import job_app

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig


@job_app.command("alloc")
def job_alloc(  # noqa: PLR0913
    ctx: typer.Context,
    account: str | None = typer.Option(None, help="与作业关联的账户"),
    argv: list[str] = typer.Option(
        None,
        help="脚本的参数。注意：总是用创建的脚本文件的路径覆盖argv[0]。如果使用了这个选项，argv[0]应该是一个一次性值。",
        show_default=False,
    ),
    begin_time: str | None = typer.Option(
        None,
        help="将作业的分配延迟到指定的时间 UNIX时间戳或时间字符串 '[MM/DD[/YY]-]HH:MM[:SS]' 或者 'infinite'",
    ),
    cpus_per_tres: str | None = typer.Option(
        None,
        help="以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的 CPU 数量（目前仅用于 gres/gpu）",
    ),
    deadline: str | None = typer.Option(
        None,
        help="作业最晚可开始的时间（UNIX 时间戳或 Slurm 识别的时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）",
    ),
    dependency: str | None = typer.Option(None, help="在本作业开始前必须满足条件的其他作业"),
    end_time: str | None = typer.Option(
        None,
        help="预期结束时间（UNIX 时间戳或时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）",
    ),
    environment: list[str] = typer.Option(None, help="要为作业设置的环境变量", show_default=False),
    group_id: str | None = typer.Option(None, help="作业所属用户的组 ID"),
    immediate: bool | None = typer.Option(None, help="若为 True，则在指定时间内资源不可用时退出"),
    job_id: int | None = typer.Option(None, help="作业 ID"),
    kill_on_node_fail: bool | None = typer.Option(None, help="若为 True，当节点故障时杀死作业"),
    memory_per_tres: str | None = typer.Option(
        None,
        help="以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的内存（MB）（目前仅用于 gres/gpu）",
    ),
    name: str | None = typer.Option(None, help="作业名称"),
    tasks: int | None = typer.Option(None, help="任务数量"),
    open_mode: list[V0043JobDescMsgOpenModeItem] | None = typer.Option(
        None,
        help="stdout/stderr 文件的打开模式",
        show_default=False,
    ),
    partition: str | None = typer.Option(None, help="作业所属分区"),
    hold: bool | None = typer.Option(None, help="暂停 (true) or 继续 (false) 任务"),
    priority: int | None = typer.Option(None, help="任务优先级"),
    qos: str | None = typer.Option(None, help="作业分配的 QoS（暂无，都是默认 Qos）"),
    requeue: bool | None = typer.Option(None, help="是否允许作业被重新排队"),
    shared: list[V0043JobDescMsgSharedItem] | None = typer.Option(
        None,
        help="作业与其他作业共享资源的方式（如允许）",
        show_default=False,
    ),
    time_limit: int | None = typer.Option(None, help="最大运行时间，单位为分钟，整数"),
    time_minimum: int | None = typer.Option(None, help="最小运行时间，单位为分钟，整数"),
    tres_per_job: str | None = typer.Option(
        None,
        help="每个作业分配的 TRES=# 列表，目前只用于 gres/gpu，比如 gres/gpu=1",
    ),
    tres_per_task: str | None = typer.Option(
        None,
        help="每个任务分配的 TRES=# 列表（逗号分隔），目前只用于 gres/gpu，比如 gres/gpu=1",
    ),
    user_id: str | None = typer.Option(None, help="作业所属用户的 UID"),
    kill_warning_flags: list[V0043JobDescMsgKillWarningFlagsItem] | None = typer.Option(
        None,
        help="与作业信号相关的标志，用于区分需要接受哪些信号",
        show_default=False,
    ),
    kill_warning_signal: str | None = typer.Option(
        None,
        help='接近结束时间时发送的信号（如 "10" 或 "USR1"）',
    ),
    current_working_directory: str = typer.Option(
        None,
        "--chdir",
        help="作业使用的工作目录",
    ),
    cpus_per_task: int | None = typer.Option(None, help="每个任务需要的 CPU 数"),
    minimum_cpus: int | None = typer.Option(None, help="所需 CPU 最小值"),
    maximum_cpus: int | None = typer.Option(None, help="所需 CPU 最大值"),
    ntasks_per_tres: int | None = typer.Option(None, help="可访问每个 GPU 的任务数"),
    memory_per_cpu: int | None = typer.Option(None, help="每个 CPU 分配的内存"),
    temporary_disk_per_node: int | None = typer.Option(None, help="每节点所需的最小临时磁盘空间"),
    standard_error: str | None = typer.Option(None, help="stderr 文件路径"),
    standard_input: str | None = typer.Option(None, help="stdin 文件路径"),
    standard_output: str | None = typer.Option(None, help="stdout 文件路径"),
    script: str | None = typer.Option(None, help="作业批处理脚本 *.toml（只会使用其中的配置项）"),
) -> None:
    """预分配作业环境"""
    with PrintProgress():
        job = job_desc_from_toml(script) if script is not None else V0043JobDescMsg()
        if account is not None:
            job.account = account
        if argv is not None:
            job.argv = argv
        if cpus_per_tres is not None:
            job.cpus_per_tres = cpus_per_tres
        if begin_time is not None:
            job.begin_time = parse_time_type(begin_time)
        if deadline is not None:
            job.deadline = parse_time_type(deadline).number
        if dependency is not None:
            job.dependency = dependency
        if end_time is not None:
            job.end_time = parse_time_type(end_time).number

        if environment is not None:
            job.environment = environment
        if not job.environment:
            job.environment = ["_THERE_MUST_BE_A_ENV_VAR_=THIS_IS_A_BUG"]

        if group_id is not None:
            job.group_id = group_id
        if immediate is not None:
            job.immediate = immediate
        if job_id is not None:
            job.job_id = job_id
        if kill_on_node_fail is not None:
            job.kill_on_node_fail = kill_on_node_fail
        if memory_per_tres is not None:
            job.memory_per_tres = memory_per_tres
        if name is not None:
            job.name = name
        if tasks is not None:
            job.tasks = tasks
        if open_mode is not None:
            job.open_mode = open_mode
        if partition is not None:
            job.partition = partition
        if hold is not None:
            job.hold = hold
        if priority is not None:
            job.priority = parse_noval_ui32(str(priority))
        if qos is not None:
            job.qos = qos
        if requeue is not None:
            job.requeue = requeue
        if shared is not None:
            job.shared = shared
        if time_limit is not None:
            job.time_limit = parse_noval_ui32(str(time_limit))
        if time_minimum is not None:
            job.time_minimum = parse_noval_ui32(str(time_minimum))
        if tres_per_job is not None:
            job.tres_per_job = tres_per_job
        if tres_per_task is not None:
            job.tres_per_task = tres_per_task
        if user_id is not None:
            job.user_id = user_id
        if kill_warning_flags is not None:
            job.kill_warning_flags = kill_warning_flags
        if kill_warning_signal is not None:
            job.kill_warning_signal = kill_warning_signal

        if current_working_directory is not None:
            job.current_working_directory = current_working_directory
        if not job.current_working_directory:
            raise typer.BadParameter("请指定工作目录 --chdir 或 'current_working_directory'")

        if cpus_per_task is not None:
            job.cpus_per_task = cpus_per_task
        if minimum_cpus is not None:
            job.minimum_cpus = minimum_cpus
        if maximum_cpus is not None:
            job.maximum_cpus = maximum_cpus

        # 默认所有 task 共享 GPU
        if ntasks_per_tres is not None:
            job.ntasks_per_tres = ntasks_per_tres
        if not job.ntasks_per_tres:
            job.ntasks_per_tres = job.tasks

        if memory_per_cpu is not None:
            job.memory_per_cpu = parse_noval_ui64(str(memory_per_cpu))
        if temporary_disk_per_node is not None:
            job.temporary_disk_per_node = temporary_disk_per_node
        if standard_error is not None:
            job.standard_error = standard_error
        if standard_input is not None:
            job.standard_input = standard_input
        if standard_output is not None:
            job.standard_output = standard_output

        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]

        print_debug(str(job.to_dict()), title="Payload", debug=debug)

        jobs = V0043JobAllocReq(job=job)

        c = SlurmClient(cfg, token, debug=debug)
        data = c.alloc(body=jobs)
        errors = data.errors
        warnings = data.warnings
        print_client_we(warnings=warnings, errors=errors)
        res_job_id = data.job_id
        res_job_submit_user_msg = data.job_submit_user_msg

        if type(res_job_id) is not int:
            res_job_id = None
        if type(res_job_submit_user_msg) is not str:
            res_job_submit_user_msg = None

    print_json_ex(
        "操作结果",
        data={
            "result": {
                "job_id": res_job_id,
                "job_submit_user_msg": res_job_submit_user_msg,
            },
        },
        key_priority=["result"],
        expand=True,
        show_raw=debug,
        annotations=submit_ann,
        show_side_notes_for_tables=True,
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
    )
