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

from ..ann import submit_ann  # noqa: TID252

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

job_app = typer.Typer(help="作业提交/查看/控制")


# @job_app.command("alloc")
# def job_alloc(  # noqa: PLR0913
#     ctx: typer.Context,
#     # 账务 / 分区 / QOS
#     account: Annotated[str | None, typer.Option(..., "--account", help="账务账户")] = None,
#     partition: Annotated[str | None, typer.Option(..., "--partition", help="分区")] = None,
#     qos: Annotated[str | None, typer.Option(..., "--qos", help="QOS")] = None,
#     # 资源与时间
#     time_limit: Annotated[str | None, typer.Option(..., "--time", help="时限，如 01:00:00")] = None,
#     nodes: Annotated[
#         str | None,
#         typer.Option(..., "--nodes", help="节点数或范围（如 1 或 1-2）"),
#     ] = None,
#     ntasks: Annotated[int | None, typer.Option(..., "--ntasks", help="任务数（tasks）")] = None,
#     cpus_per_task: Annotated[
#         int | None,
#         typer.Option(..., "--cpus-per-task", help="每个任务的CPU核数"),
#     ] = None,
#     gpus: Annotated[
#         int | None,
#         typer.Option(..., "--gpus", help="每节点GPU数（自动映射为 gres/gpu:N）"),
#     ] = None,
#     mem: Annotated[int | None, typer.Option(..., "--mem", help="每节点内存（MiB）")] = None,
#     # 约束与排队
#     constraint: Annotated[
#         str | None,
#         typer.Option(
#             ...,
#             "--constraint",
#             help="节点特性约束（如 a100|h100）",
#         ),
#     ] = None,
#     exclude: Annotated[
#         str | None,
#         typer.Option(..., "--exclude", help="排除节点，逗号分隔"),
#     ] = None,
#     reservation: Annotated[
#         str | None,
#         typer.Option(..., "--reservation", help="使用预留名"),
#     ] = None,
#     dependency: Annotated[
#         str | None,
#         typer.Option(
#             ...,
#             "--dependency",
#             help="依赖（如 afterok:12345）",
#         ),
#     ] = None,
#     begin_time: Annotated[
#         str | None,
#         typer.Option(
#             ...,
#             "--begin",
#             help="延迟开始时间",
#         ),
#     ] = None,
#     # I/O 与目录
#     name: Annotated[str | None, typer.Option(..., "--name", help="作业名")] = None,
#     chdir: Annotated[str | None, typer.Option(..., "--chdir", help="作业工作目录")] = None,
#     output: Annotated[
#         str | None,
#         typer.Option(
#             ...,
#             "--output",
#             help="标准输出路径（如 /path/slurm-%j.out）",
#         ),
#     ] = None,
#     error: Annotated[
#         str | None,
#         typer.Option(
#             ...,
#             "--error",
#             help="标准错误路径（如 /path/slurm-%j.err）",
#         ),
#     ] = None,
#     # 通知
#     mail_user: Annotated[
#         str | None,
#         typer.Option(..., "--mail-user", help="邮件通知收件人（暂不可用）", hidden=True),
#     ] = None,
#     mail_type: Annotated[
#         list[str] | None,
#         typer.Option(
#             ...,
#             "--mail-type",
#             help="邮件通知类型，可多次传入（如 --mail-type END --mail-type FAIL）",
#             hide_input=True,
#         ),
#     ] = None,
#     # 其他信息
#     comment: Annotated[
#         str | None,
#         typer.Option(..., "--comment", help="用户提交的注释信息"),
#     ] = None,
#     # 环境变量（可多次传入 KEY=VAL）
#     env: Annotated[
#         list[str] | None,
#         typer.Option(
#             ...,
#             "--env",
#             help="附加环境变量（可多次传入，如 --env FOO=bar）",
#         ),
#     ] = None,
# ) -> None:
#     "(预)分配资源用于作业或ssh连接"

#     def _no_val(x: int | float | None) -> dict[str, int | str | float]:
#         return {
#             "set": True if x else False,
#             "infinite": True if not x else False,
#             "number": x if x else "",
#         }

#     with PrintProgress():
#         # 组装 REST 请求体（对应 v0.0.43_job_desc_msg 的字段）
#         job: dict[str, Any] = {}

#         # ——— 基础字段（名字/分区/QOS/账户） ———
#         if name:
#             job["name"] = name
#         if partition:
#             job["partition"] = partition
#         if qos:
#             job["qos"] = qos
#         if account:
#             job["account"] = account

#         # ——— 资源与时间 ———
#         # nodes：REST 支持范围字符串（如 "1" 或 "1-2"）
#         if nodes:
#             job["nodes"] = str(nodes)
#         if ntasks is not None:
#             job["tasks"] = ntasks  # REST 字段是 tasks（非 ntasks）
#         if cpus_per_task is not None:
#             job["cpus_per_task"] = cpus_per_task
#         if gpus is not None:
#             job["tres_per_node"] = f"gres/gpu:{gpus}"  # 通用 GPU 申请写法（TRES/GRES）
#         if mem is not None:
#             job["memory_per_node"] = mem  # 单位 MiB；部分站点也接受 GiB 需按站点约定
#         if time_limit:
#             job["time_limit"] = time_limit  # 常见集群接受 "HH:MM:SS" 或分钟值

#         # ——— I/O 与目录 ———
#         if output:
#             job["standard_output"] = output
#         if error:
#             job["standard_error"] = error
#         if chdir:
#             job["current_working_directory"] = chdir

#         # ——— 约束/排除/预留/依赖/延时 ———
#         if constraint:
#             job["constraints"] = constraint
#         if exclude:
#             job["excluded_nodes"] = exclude
#         if reservation:
#             job["reservation"] = reservation
#         if dependency:
#             job["dependency"] = dependency
#         if begin_time:
#             job["begin_time"] = begin_time

#         # ——— 邮件通知 ———
#         if mail_user:
#             job["mail_user"] = mail_user
#         if mail_type:
#             job["mail_type"] = mail_type  # 例如 ["END","FAIL"]

#         # ——— 其他信息 ———
#         if comment:
#             job["comment"] = comment

#         # ——— 环境变量 ———
#         job["environment"] = []
#         if env:
#             env_map: dict[str, str] = {}
#             for kv in env:
#                 if "=" not in kv:
#                     print_error(f"--env 需要 KEY=VAL 形式，收到：{kv}")
#                     raise typer.Exit(code=2)
#                 k, v = kv.split("=", 1)
#                 env_map[k] = v
#             job["environment"] = [f"{k}={v}" for k, v in env_map.items()]

#         job["environment"].append("_THERE_MUST_BE_A_ENV_VAR_=THIS_IS_A_BUG")

#         req: dict[str, Any] = {"job": job}

#         cfg: AppConfig = ctx.obj["cfg"]
#         token: str = ctx.obj["token"]
#         debug: bool = ctx.obj["debug"]
#         c = SlurmClient(cfg, token, debug=debug)

#         # POST /slurm/v0.0.43/job/allocate
#         resp = c.post_json("/job/allocate", body=req)

#     print_debug(resp, "原始数据", debug=debug)
#     if resp.get("errors"):
#         print_error("提交失败")
#         print_kv("错误", resp["errors"], cfg.logging.dict_style)
#     else:
#         del resp["errors"]
#         del resp["warnings"]
#         del resp["meta"]

#         print_json_ex(
#             "提交结果",
#             data={"result": resp},
#             key_priority=["result"],
#             expand=True,
#             show_raw=debug,
#             annotations=submit_ann,
#             show_side_notes_for_tables=True,
#             notes_panel_title="注释",
#             show_side_notes_for_dicts=True,
#             dict_notes_min_hits=2,
#             dict_notes_max_depth=3,
#             dict_notes_panel_title="相关信息",
#         )


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
        notes_panel_title="注释",
        show_side_notes_for_dicts=True,
        dict_notes_min_hits=2,
        dict_notes_max_depth=3,
        dict_notes_panel_title="相关信息",
    )
