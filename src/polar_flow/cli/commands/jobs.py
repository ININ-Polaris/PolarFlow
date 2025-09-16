from typing import TYPE_CHECKING, Any

import typer

from polar_flow.cli.client import SlurmClient
from polar_flow.cli.printers import print_error, print_info, print_kv

if TYPE_CHECKING:
    from polar_flow.cli.config import AppConfig

job_app = typer.Typer(help="作业提交/查看/控制")


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    state: str | None = typer.Option(None, help="筛选状态，如 RUNNING,PENDING 等"),
    detail: bool = typer.Option(False, help="详细输出，会打印所有获取到的信息"),
) -> None:
    """列出作业（/jobs/ 或 /jobs/state/）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get("/jobs/state/") if state else c.get("/jobs/")

    if detail:
        title = f"作业列表（状态={state}）" if state else "作业列表"
        print_kv(title, data, cfg.logging.dict_style)



@job_app.command("show")
def job_show(ctx: typer.Context, job_id: int = typer.Argument(..., help="作业 ID")) -> None:
    """查看单个作业详情（/job/{job_id}）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    data = c.get(f"/job/{job_id}")
    print_kv(f"作业 {job_id}", data, cfg.logging.dict_style)


@job_app.command("submit")
def job_submit(  # noqa: PLR0913
    ctx: typer.Context,
    script: str | None = typer.Option(None, "--script", help="脚本内容（直接传入）"),
    script_path: str | None = typer.Option(None, "--file", help="脚本路径（读取后提交）"),
    partition: str | None = typer.Option(None, "--partition", help="分区"),
    qos: str | None = typer.Option(None, "--qos", help="QOS"),
    account: str | None = typer.Option(None, "--account", help="账务账户"),
    time_limit: str | None = typer.Option(None, "--time", help="时限，如 01:00:00"),
    nodes: int | None = typer.Option(None, "--nodes", help="节点数"),
    ntasks: int | None = typer.Option(None, "--ntasks", help="任务数"),
) -> None:
    """提交新作业（POST /job/submit）"""
    if not script and not script_path:
        print_error("请使用 --script 或 --file 提供作业脚本内容")
        raise typer.Exit(code=2)
    if script_path and not script:
        try:
            with open(script_path, encoding="utf-8") as file:
                script = file.read()
        except Exception as e:  # noqa: BLE001
            print_error(f"读取脚本失败：{e}")
            raise typer.Exit(code=1) from None

    req: dict[str, Any] = {"script": script}
    # 可根据需要补充更多请求字段
    opts = {
        "partition": partition,
        "qos": qos,
        "account": account,
        "time_limit": time_limit,
        "nodes": nodes,
        "ntasks": ntasks,
    }
    req.update({k: v for k, v in opts.items() if v is not None})

    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    resp = c.post_json("/job/submit", body=req)  # POST /slurm/v0.0.43/job/submit
    if resp.get("errors"):
        print_error("提交失败")
        print_kv("错误", resp["errors"], cfg.logging.dict_style)
    else:
        print_info("提交成功")
        print_kv("结果", resp, cfg.logging.dict_style)


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
    """取消/信号作业（DELETE /job/{job_id}，可带 ?signal=）"""
    cfg: AppConfig = ctx.obj["cfg"]
    token: str = ctx.obj["token"]
    debug: bool = ctx.obj["debug"]
    c = SlurmClient(cfg, token, debug=debug)
    path = f"/job/{job_id}" + (f"?signal={signal}" if signal else "")
    resp = c.delete(path)
    print_kv("操作结果", resp, cfg.logging.dict_style)
