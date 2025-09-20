from __future__ import annotations

from pathlib import Path
import tomllib
from collections.abc import Iterable, Mapping
from typing import Any

# 所有支持的键
_SUPPORTED_KEYS = {
    # str 与作业关联的账户
    "account",
    # int 作业 ID
    "job_id",
    # str 作业所属分区
    "partition",
    # str 作业名称
    "name",
    # int 任务数量
    "tasks",
    # str 作业使用的工作目录
    "current_working_directory",
    # str 每个作业分配的 TRES=# 列表，目前只用于 gres/gpu，比如 gres/gpu=1
    "tres_per_job",
    # str 每个任务分配的 TRES=# 列表（逗号分隔），目前只用于 gres/gpu，比如 gres/gpu=1
    "tres_per_task",
    # str 以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的 CPU 数量（目前仅用于 gres/gpu）
    "cpus_per_tres",
    # int 每个任务需要的 CPU 数
    "cpus_per_task",
    # int 所需 CPU 最小值
    "minimum_cpus",
    # int 所需 CPU 最大值
    "maximum_cpus",
    # str 以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的内存（MB）（目前仅用于 gres/gpu）
    "memory_per_tres",
    # int 可访问每个 GPU 的任务数
    "ntasks_per_tres",
    # int 每个 CPU 分配的内存
    "memory_per_cpu",
    # int 每节点所需的最小临时磁盘空间
    "temporary_disk_per_node",
    # str 作业所属用户的 UID
    "user_id",
    # str 作业所属用户的组 ID
    "group_id",
    # list[str] 脚本的参数。注意：总是用创建的脚本文件的路径覆盖argv[0]。如果使用了这个选项，argv[0]应该是一个一次性值。
    "argv",
    # list[str] 要为作业设置的环境变量
    "environment",
    # bool 若为 True，则在指定时间内资源不可用时退出
    "immediate",
    # str 将作业的分配延迟到指定的时间 UNIX时间戳或时间字符串 '[MM/DD[/YY]-]HH:MM[:SS]' 或者 'infinite'
    "begin_time",
    # str 作业最晚可开始的时间（UNIX 时间戳或 Slurm 识别的时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）
    "deadline",
    # str 在本作业开始前必须满足条件的其他作业
    "dependency",
    # str 预期结束时间（UNIX 时间戳或时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）
    "end_time",
    # int 最大运行时间，单位为分钟，整数
    "time_limit",
    # int 最小运行时间，单位为分钟，整数
    "time_minimum",
    # bool 若为 True，当节点故障时杀死作业
    "kill_on_node_fail",
    # bool 暂停 (true) or 继续 (false) 任务
    "hold",
    # int 优先级
    "priority",
    # str 作业分配的 QoS（暂无，都是默认 Qos）
    "qos",
    # bool 是否允许作业被重新排队
    "requeue",
    # list[...] 与作业信号相关的标志，用于区分需要接受哪些信号
    # ARRAY_TASK, BATCH_JOB, CRON_JOBS, FEDERATION_REQUEUE, FULL_JOB,
    # FULL_STEPS_ONLY, HURRY, NO_SIBLING_JOBS, OUT_OF_MEMORY,
    # RESERVATION_JOB, VERBOSE, WARNING_SENT
    "kill_warning_flags",
    # str 接近结束时间时发送的信号（如 "10" 或 "USR1"）
    "kill_warning_signal",
    # str stderr 文件路径
    "standard_error",
    # str stdin 文件路径
    "standard_input",
    # str stdout 文件路径
    "standard_output",
}


def _as_list(maybe: Any) -> list[str]:
    """把 str | list[str] 统一成 list[str]（为空时返回 []）。"""
    if maybe is None:
        return []
    if isinstance(maybe, str):
        return [maybe]
    if isinstance(maybe, Iterable):
        # 只保留字符串项
        return [str(x) for x in maybe]
    return [str(maybe)]


def _get_script_from_toml_section(section: Mapping[str, Any]) -> str:
    """
    根据 [script] 的约定生成脚本文件:
      - 可选 shell: str（写 shebang，用作解释器）
      - 可选 pre: str | list[str]
      - 可选 command: str | list[str]
    将 pre + command 依次用 '\n'.join() 拼接；若设置了 shell，前面加 shebang。
    返回脚本文件的绝对路径（已设置可执行位）。
    """
    shell = section.get("shell")
    pre_lines = _as_list(section.get("pre"))
    cmd_lines = _as_list(section.get("command"))

    lines: list[str] = []
    if shell:
        # 使用 env 以获得 PATH 解析；兼容 zsh/bash/sh
        lines.append(f"#!/usr/bin/env {shell}")
    # 拼接主体
    lines.extend(pre_lines)
    lines.extend(cmd_lines)
    return "\n".join(lines).rstrip() + "\n"


def job_submit_from_toml(toml_path: str) -> dict[str, Any]:
    """
    从 TOML 文件读取参数，构造与原 @job_app.command(\"submit\") 等价的 payload。
    仅当键出现且值非 None 时写入 payload；[script] 额外规则见上。
    """
    with open(toml_path, "rb") as fp:
        cfg = tomllib.load(fp)  # 返回 dict

    payload: dict[str, Any] = {}

    # 1) 复制与原函数同名的顶层键
    for key in _SUPPORTED_KEYS:
        if key in cfg and cfg[key] is not None and cfg[key] != "":
            payload[key] = cfg[key]

    # 2) 特殊处理 [script]
    script_cfg = cfg.get("script")
    if isinstance(script_cfg, Mapping):
        payload["script"] = _get_script_from_toml_section(script_cfg)
    elif isinstance(script_cfg, str) and script_cfg.startswith("file://"):
        path_str = script_cfg[len("file://") :]
        path = Path(path_str)
        if path.exists() and path.is_file():
            with open(path, encoding="utf-8") as f:
                payload["script"] = f.read()
        else:
            raise FileNotFoundError(f"未找到脚本文件: {path_str}")

    # 3) 显式的布尔字段保持布尔
    for b in (
        "contiguous",
        "immediate",
        "kill_on_node_fail",
        "overcommit",
        "hold",
        "requeue",
        "wait_all_nodes",
    ):
        if b in payload and payload[b] is not None:
            payload[b] = bool(payload[b])

    # 4) 返回结构
    return payload
