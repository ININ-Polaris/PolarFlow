from typing import TYPE_CHECKING, Annotated, Any

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import PrintProgress, print_error, print_info, print_json_ex, print_kv

from .ann import job_info_ann, submit_ann

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

job_app = typer.Typer(help="作业提交/查看/控制")


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    vvv: Annotated[bool, typer.Option(..., "--vvv", help="列出完整信息（会非常多）")] = False,
) -> None:
    """列出作业"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get("/jobs/state/")

    title = "作业状态"
    print_json_ex(
        title,
        data={"jobs": data["jobs"]},
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
    )

    if vvv:
        with PrintProgress("[bold cyan]请稍候，正在获取详细信息...[/]"):
            data_detail = c.get("/jobs/")
        title = "作业列表"
        print_json_ex(
            title,
            data={"jobs": data_detail["jobs"]},
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
    vvv: Annotated[bool, typer.Option(..., "--vvv", help="列出完整信息（会非常多）")] = False,
) -> None:
    """查看单个作业详情（/job/{job_id}）"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        data = c.get(f"/job/{job_id}")

    jobs = data.get("jobs", [])

    KEEP_KEYS = {
        "command",
        "current_working_directory",
        "flags",
        "job_id",
        "hold",
        "group_id",
        "group_name",
        "job_state",
        "state_reason",
        "tres_req_str",
        "name",
        "user",
        "account",
        "partition",
        "qos",
    }

    filtered_jobs = jobs
    max_cols = 8
    if not vvv:
        filtered_jobs = []
        max_cols = 4
        for job in jobs:
            filtered_job = {k: job[k] for k in KEEP_KEYS if k in job}
            filtered_jobs.append(filtered_job)

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


@job_app.command("submit")
def job_submit(  # noqa: PLR0913
    ctx: typer.Context,
    # 彻底禁止直接传脚本内容
    script_path: str = typer.Argument(help="脚本路径（从文件读取后提交）"),
    # 账务 / 分区 / QOS
    account: Annotated[str | None, typer.Option(..., "--account", help="账务账户")] = None,
    partition: Annotated[str | None, typer.Option(..., "--partition", help="分区")] = None,
    qos: Annotated[str | None, typer.Option(..., "--qos", help="QOS")] = None,
    # 资源与时间
    time_limit: Annotated[str | None, typer.Option(..., "--time", help="时限，如 01:00:00")] = None,
    nodes: Annotated[
        str | None,
        typer.Option(..., "--nodes", help="节点数或范围（如 1 或 1-2）"),
    ] = None,
    ntasks: Annotated[int | None, typer.Option(..., "--ntasks", help="任务数（tasks）")] = None,
    cpus_per_task: Annotated[
        int | None,
        typer.Option(..., "--cpus-per-task", help="每个任务的CPU核数"),
    ] = None,
    gpus: Annotated[
        int | None,
        typer.Option(..., "--gpus", help="每节点GPU数（自动映射为 gres/gpu:N）"),
    ] = None,
    mem: Annotated[int | None, typer.Option(..., "--mem", help="每节点内存（MiB）")] = None,
    # 约束与排队
    constraint: Annotated[
        str | None,
        typer.Option(
            ...,
            "--constraint",
            help="节点特性约束（如 a100|h100）",
        ),
    ] = None,
    exclude: Annotated[
        str | None,
        typer.Option(..., "--exclude", help="排除节点，逗号分隔"),
    ] = None,
    reservation: Annotated[
        str | None,
        typer.Option(..., "--reservation", help="使用预留名"),
    ] = None,
    dependency: Annotated[
        str | None,
        typer.Option(
            ...,
            "--dependency",
            help="依赖（如 afterok:12345）",
        ),
    ] = None,
    begin_time: Annotated[
        str | None,
        typer.Option(
            ...,
            "--begin",
            help="延迟开始时间（如 now+10min 或 23:00）",
        ),
    ] = None,
    # I/O 与目录
    name: Annotated[str | None, typer.Option(..., "--name", help="作业名")] = None,
    chdir: Annotated[str | None, typer.Option(..., "--chdir", help="作业工作目录")] = None,
    output: Annotated[
        str | None,
        typer.Option(
            ...,
            "--output",
            help="标准输出路径（如 /path/slurm-%j.out）",
        ),
    ] = None,
    error: Annotated[
        str | None,
        typer.Option(
            ...,
            "--error",
            help="标准错误路径（如 /path/slurm-%j.err）",
        ),
    ] = None,
    # 通知
    mail_user: Annotated[
        str | None,
        typer.Option(..., "--mail-user", help="邮件通知收件人（暂不可用）"),
    ] = None,
    mail_type: Annotated[
        list[str] | None,
        typer.Option(
            ...,
            "--mail-type",
            help="邮件通知类型，可多次传入（如 --mail-type END --mail-type FAIL）",
        ),
    ] = None,
    # 其他信息
    comment: Annotated[
        str | None,
        typer.Option(..., "--comment", help="用户提交的注释信息"),
    ] = None,
    # 环境变量（可多次传入 KEY=VAL）
    env: Annotated[
        list[str] | None,
        typer.Option(
            ...,
            "--env",
            help="附加环境变量（可多次传入，如 --env FOO=bar）",
        ),
    ] = None,
) -> None:
    """提交新作业（POST /slurm/v0.0.43/job/submit）"""
    with PrintProgress():
        # 读取脚本
        try:
            with open(script_path, encoding="utf-8") as f:
                script_text = f.read()
        except Exception as e:  # noqa: BLE001
            print_error(f"读取脚本失败：{e}")
            raise typer.Exit(code=1) from None

        # 组装 REST 请求体（对应 v0.0.43_job_desc_msg 的字段）
        job: dict[str, Any] = {"script": script_text}

        # ——— 基础字段（名字/分区/QOS/账户） ———
        if name:
            job["name"] = name
        if partition:
            job["partition"] = partition
        if qos:
            job["qos"] = qos
        if account:
            job["account"] = account

        # ——— 资源与时间 ———
        # nodes：REST 支持范围字符串（如 "1" 或 "1-2"）
        if nodes:
            job["nodes"] = str(nodes)
        if ntasks is not None:
            job["tasks"] = ntasks  # REST 字段是 tasks（非 ntasks）
        if cpus_per_task is not None:
            job["cpus_per_task"] = cpus_per_task
        if gpus is not None:
            job["tres_per_node"] = f"gres/gpu:{gpus}"  # 通用 GPU 申请写法（TRES/GRES）
        if mem is not None:
            job["memory_per_node"] = mem  # 单位 MiB；部分站点也接受 GiB 需按站点约定
        if time_limit:
            job["time_limit"] = time_limit  # 常见集群接受 "HH:MM:SS" 或分钟值

        # ——— I/O 与目录 ———
        if output:
            job["standard_output"] = output
        if error:
            job["standard_error"] = error
        if chdir:
            job["current_working_directory"] = chdir

        # ——— 约束/排除/预留/依赖/延时 ———
        if constraint:
            job["constraints"] = constraint
        if exclude:
            job["excluded_nodes"] = exclude
        if reservation:
            job["reservation"] = reservation
        if dependency:
            job["dependency"] = dependency
        if begin_time:
            job["begin_time"] = begin_time

        # ——— 邮件通知 ———
        if mail_user:
            job["mail_user"] = mail_user
        if mail_type:
            job["mail_type"] = mail_type  # 例如 ["END","FAIL"]

        # ——— 其他信息 ———
        if comment:
            job["comment"] = comment

        # ——— 环境变量 ———
        job["environment"] = []
        if env:
            env_map: dict[str, str] = {}
            for kv in env:
                if "=" not in kv:
                    print_error(f"--env 需要 KEY=VAL 形式，收到：{kv}")
                    raise typer.Exit(code=2)
                k, v = kv.split("=", 1)
                env_map[k] = v
            job["environment"] = [f"{k}={v}" for k, v in env_map.items()]

        job["environment"].append("_THERE_MUST_BE_A_ENV_VAR_=THIS_IS_A_BUG")

        req: dict[str, Any] = {"job": job}

        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)

        # POST /slurm/v0.0.43/job/submit
        resp = c.post_json("/job/submit", body=req)

    if resp.get("errors"):
        print_error("提交失败")
        print_kv("错误", resp["errors"], cfg.logging.dict_style)
    else:
        # print_kv("结果", resp, cfg.logging.dict_style)
        del resp["errors"]
        del resp["warnings"]
        del resp["meta"]

        print_json_ex(
            "提交结果",
            data={"result": resp},
            key_priority=["result"],
            expand=True,
            show_raw=debug,
            annotations=submit_ann,
            show_side_notes_for_tables=True,
            notes_panel_title="注释",
            show_side_notes_for_dicts=True,
            dict_notes_min_hits=2,
            dict_notes_max_depth=3,
            dict_notes_panel_title="相关信息",
        )


@job_app.command("alloc")
def job_alloc(  # noqa: PLR0913
    ctx: typer.Context,
    # 账务 / 分区 / QOS
    account: Annotated[str | None, typer.Option(..., "--account", help="账务账户")] = None,
    partition: Annotated[str | None, typer.Option(..., "--partition", help="分区")] = None,
    qos: Annotated[str | None, typer.Option(..., "--qos", help="QOS")] = None,
    # 资源与时间
    time_limit: Annotated[str | None, typer.Option(..., "--time", help="时限，如 01:00:00")] = None,
    nodes: Annotated[
        str | None,
        typer.Option(..., "--nodes", help="节点数或范围（如 1 或 1-2）"),
    ] = None,
    ntasks: Annotated[int | None, typer.Option(..., "--ntasks", help="任务数（tasks）")] = None,
    cpus_per_task: Annotated[
        int | None,
        typer.Option(..., "--cpus-per-task", help="每个任务的CPU核数"),
    ] = None,
    gpus: Annotated[
        int | None,
        typer.Option(..., "--gpus", help="每节点GPU数（自动映射为 gres/gpu:N）"),
    ] = None,
    mem: Annotated[int | None, typer.Option(..., "--mem", help="每节点内存（MiB）")] = None,
    # 约束与排队
    constraint: Annotated[
        str | None,
        typer.Option(
            ...,
            "--constraint",
            help="节点特性约束（如 a100|h100）",
        ),
    ] = None,
    exclude: Annotated[
        str | None,
        typer.Option(..., "--exclude", help="排除节点，逗号分隔"),
    ] = None,
    reservation: Annotated[
        str | None,
        typer.Option(..., "--reservation", help="使用预留名"),
    ] = None,
    dependency: Annotated[
        str | None,
        typer.Option(
            ...,
            "--dependency",
            help="依赖（如 afterok:12345）",
        ),
    ] = None,
    begin_time: Annotated[
        str | None,
        typer.Option(
            ...,
            "--begin",
            help="延迟开始时间（如 now+10min 或 23:00）",
        ),
    ] = None,
    # I/O 与目录
    name: Annotated[str | None, typer.Option(..., "--name", help="作业名")] = None,
    chdir: Annotated[str | None, typer.Option(..., "--chdir", help="作业工作目录")] = None,
    output: Annotated[
        str | None,
        typer.Option(
            ...,
            "--output",
            help="标准输出路径（如 /path/slurm-%j.out）",
        ),
    ] = None,
    error: Annotated[
        str | None,
        typer.Option(
            ...,
            "--error",
            help="标准错误路径（如 /path/slurm-%j.err）",
        ),
    ] = None,
    # 通知
    mail_user: Annotated[
        str | None,
        typer.Option(..., "--mail-user", help="邮件通知收件人（暂不可用）"),
    ] = None,
    mail_type: Annotated[
        list[str] | None,
        typer.Option(
            ...,
            "--mail-type",
            help="邮件通知类型，可多次传入（如 --mail-type END --mail-type FAIL）",
        ),
    ] = None,
    # 其他信息
    comment: Annotated[
        str | None,
        typer.Option(..., "--comment", help="用户提交的注释信息"),
    ] = None,
    # 环境变量（可多次传入 KEY=VAL）
    env: Annotated[
        list[str] | None,
        typer.Option(
            ...,
            "--env",
            help="附加环境变量（可多次传入，如 --env FOO=bar）",
        ),
    ] = None,
) -> None:
    "(预)分配资源用于作业或ssh连接"
    with PrintProgress():
        # 组装 REST 请求体（对应 v0.0.43_job_desc_msg 的字段）
        job: dict[str, Any] = {}

        # ——— 基础字段（名字/分区/QOS/账户） ———
        if name:
            job["name"] = name
        if partition:
            job["partition"] = partition
        if qos:
            job["qos"] = qos
        if account:
            job["account"] = account

        # ——— 资源与时间 ———
        # nodes：REST 支持范围字符串（如 "1" 或 "1-2"）
        if nodes:
            job["nodes"] = str(nodes)
        if ntasks is not None:
            job["tasks"] = ntasks  # REST 字段是 tasks（非 ntasks）
        if cpus_per_task is not None:
            job["cpus_per_task"] = cpus_per_task
        if gpus is not None:
            job["tres_per_node"] = f"gres/gpu:{gpus}"  # 通用 GPU 申请写法（TRES/GRES）
        if mem is not None:
            job["memory_per_node"] = mem  # 单位 MiB；部分站点也接受 GiB 需按站点约定
        if time_limit:
            job["time_limit"] = time_limit  # 常见集群接受 "HH:MM:SS" 或分钟值

        # ——— I/O 与目录 ———
        if output:
            job["standard_output"] = output
        if error:
            job["standard_error"] = error
        if chdir:
            job["current_working_directory"] = chdir

        # ——— 约束/排除/预留/依赖/延时 ———
        if constraint:
            job["constraints"] = constraint
        if exclude:
            job["excluded_nodes"] = exclude
        if reservation:
            job["reservation"] = reservation
        if dependency:
            job["dependency"] = dependency
        if begin_time:
            job["begin_time"] = begin_time

        # ——— 邮件通知 ———
        if mail_user:
            job["mail_user"] = mail_user
        if mail_type:
            job["mail_type"] = mail_type  # 例如 ["END","FAIL"]

        # ——— 其他信息 ———
        if comment:
            job["comment"] = comment

        # ——— 环境变量 ———
        job["environment"] = []
        if env:
            env_map: dict[str, str] = {}
            for kv in env:
                if "=" not in kv:
                    print_error(f"--env 需要 KEY=VAL 形式，收到：{kv}")
                    raise typer.Exit(code=2)
                k, v = kv.split("=", 1)
                env_map[k] = v
            job["environment"] = [f"{k}={v}" for k, v in env_map.items()]

        job["environment"].append("_THERE_MUST_BE_A_ENV_VAR_=THIS_IS_A_BUG")

        req: dict[str, Any] = {"job": job}

        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)

        # POST /slurm/v0.0.43/job/allocate
        resp = c.post_json("/job/allocate", body=req)

    if resp.get("errors"):
        print_error("提交失败")
        print_kv("错误", resp["errors"], cfg.logging.dict_style)
    else:
        del resp["errors"]
        del resp["warnings"]
        del resp["meta"]

        print_json_ex(
            "提交结果",
            data={"result": resp},
            key_priority=["result"],
            expand=True,
            show_raw=debug,
            annotations=submit_ann,
            show_side_notes_for_tables=True,
            notes_panel_title="注释",
            show_side_notes_for_dicts=True,
            dict_notes_min_hits=2,
            dict_notes_max_depth=3,
            dict_notes_panel_title="相关信息",
        )


@job_app.command("cancel")
def job_cancel(
    ctx: typer.Context,
    job_id: int = typer.Argument(..., help="作业 ID"),
    signal: str | None = typer.Option(
        None,
        "--signal",
        help="发送信号而非直接取消，例如 TERM,KILL",
    ),
) -> None:
    """取消/信号作业"""
    with PrintProgress():
        cfg: AppConfig = ctx.obj["cfg"]
        token: str = ctx.obj["token"]
        debug: bool = ctx.obj["debug"]
        c = SlurmClient(cfg, token, debug=debug)
        path = f"/job/{job_id}" + (f"?signal={signal}" if signal else "")
        resp = c.delete(path)

    status = resp.get("status")
    if status:
        error: dict[str, Any] = status[0].get("error", None)
        print_error(f"失败原因：{error['message']}", f"取消失败 [{error['code']}]")
    elif status == []:
        print_info(f"已取消分配的作业，ID[{job_id}]", "取消成功")
    else:
        print_error("请联系管理员", "未知错误")
    # print_kv("操作结果", resp, cfg.logging.dict_style)
