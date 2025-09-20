from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

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
from polar_flow._vendor.slurm_client.models.v0043_uint_32_no_val_struct import (
    V0043Uint32NoValStruct,
)
from polar_flow._vendor.slurm_client.models.v0043_uint_64_no_val_struct import (
    V0043Uint64NoValStruct,
)
from polar_flow.cli.commands.jobs.utils import parse_noval_ui32, parse_noval_ui64, parse_time_type

_MAX_SCRIPT_BYTES = 20 * 1024  # 20KB 安全上限，按需调整
_HTTP_TIMEOUT = (5, 15)  # 连接超时 5s，读超时 15s


class Time(int): ...


class TimeInt(int): ...


# 所有支持的键
_SUPPORTED_KEYS: dict[str, type] = {
    "account": str,
    "job_id": int,
    "partition": str,
    "name": str,
    "tasks": int,
    "current_working_directory": str,
    "tres_per_job": str,
    "tres_per_task": str,
    "cpus_per_tres": str,
    "cpus_per_task": int,
    "minimum_cpus": int,
    "maximum_cpus": int,
    "memory_per_tres": str,
    "ntasks_per_tres": int,
    "memory_per_cpu": V0043Uint64NoValStruct,
    "temporary_disk_per_node": int,
    "user_id": str,
    "group_id": str,
    "argv": list[str],
    "environment": list[str],
    "immediate": bool,
    "begin_time": Time,
    "deadline": TimeInt,
    "dependency": str,
    "end_time": TimeInt,
    "time_limit": V0043Uint32NoValStruct,
    "time_minimum": V0043Uint32NoValStruct,
    "kill_on_node_fail": bool,
    "hold": bool,
    "priority": V0043Uint32NoValStruct,
    "qos": str,
    "requeue": bool,
    "kill_warning_signal": str,
    "standard_error": str,
    "standard_input": str,
    "standard_output": str,
}

# 单独处理的键
# kill_warning_flags V0043JobDescMsgKillWarningFlagsItem
# open_mode V0043JobDescMsgOpenModeItem
# shared V0043JobDescMsgSharedItem
# script Any


def _as_list(maybe: Any) -> list[str]:
    """把 str | list[str] 统一成 list[str]（为空时返回 []）。"""
    if maybe is None or maybe == "":
        return []
    if isinstance(maybe, str):
        return [maybe]
    if isinstance(maybe, Iterable):
        return [str(x) for x in maybe]
    raise ValueError(f"{maybe} 无法转化为列表")


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


def job_desc_from_toml(toml_path: str) -> V0043JobDescMsg:
    """
    从 TOML 文件读取参数，构造与原 @job_app.command(\"submit\") 等价的 payload。
    仅当键出现且值非 None 时写入 payload；[script] 额外规则见上。
    """
    with open(toml_path, "rb") as fp:
        cfg = tomllib.load(fp)  # 返回 dict

    job = V0043JobDescMsg()

    for key, type_ in _SUPPORTED_KEYS.items():
        if key in cfg and cfg[key] is not None and cfg[key] != "":
            try:
                if type_ in (str, int, bool):
                    setattr(job, key, type_(cfg[key]))
                elif type_ is list[str]:
                    setattr(job, key, _as_list(cfg[key]))
                elif type_ is Time:
                    setattr(job, key, parse_time_type(cfg[key]))
                elif type_ is TimeInt:
                    setattr(job, key, parse_time_type(cfg[key]).number)
                elif type_ is V0043Uint32NoValStruct:
                    setattr(job, key, parse_noval_ui32(cfg[key]))
                elif type_ is V0043Uint64NoValStruct:
                    setattr(job, key, parse_noval_ui64(cfg[key]))

            except ValueError as ve:
                raise ValueError(f"序列化 {key} 出错: {ve}") from None

    if not job.environment:
        job.environment = ["_THERE_MUST_BE_A_ENV_VAR_=THIS_IS_A_BUG"]

    if "kill_warning_flags" in cfg:
        try:
            kwf = _as_list(cfg["kill_warning_flags"])
        except ValueError as ve:
            raise ValueError(f"序列化 kill_warning_flags 出错: {ve}") from None
        job.kill_warning_flags = []
        for x in kwf:
            try:
                job.kill_warning_flags.append(V0043JobDescMsgKillWarningFlagsItem(x))
            except ValueError:
                raise ValueError(f"'{x}' 不是一个合法的 kill_warning_flags") from ValueError

    if "open_mode" in cfg:
        try:
            kwf = _as_list(cfg["open_mode"])
        except ValueError as ve:
            raise ValueError(f"序列化 open_mode 出错: {ve}") from None
        job.open_mode = []
        for x in kwf:
            try:
                job.open_mode.append(V0043JobDescMsgOpenModeItem(x))
            except ValueError:
                raise ValueError(f"'{x}' 不是一个合法的 open_mode") from ValueError

    if "shared" in cfg:
        kwf = _as_list(cfg["shared"])
        if type(kwf) is not list:
            raise ValueError(f"shared 应为列表而不是 {type(kwf).__name__}")
        job.shared = []
        for x in kwf:
            try:
                job.shared.append(V0043JobDescMsgSharedItem(x))
            except ValueError:
                raise ValueError(f"'{x}' 不是一个合法的 shared") from ValueError

    # 特殊处理 [script]
    script_cfg = cfg.get("script")
    if isinstance(script_cfg, Mapping):
        job.script = _get_script_from_toml_section(script_cfg)
    elif isinstance(script_cfg, str) and script_cfg.startswith("file://"):
        path_str = script_cfg[len("file://") :]
        path = Path(path_str)
        if path.exists() and path.is_file():
            with open(path, encoding="utf-8") as f:
                job.script = f.read()
        else:
            raise FileNotFoundError(f"未找到脚本文件: {path_str}")
    elif isinstance(script_cfg, str):
        # 增加对 remote 的支持：http(s) URL
        parsed = urlparse(script_cfg)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            try:
                # 先流式请求以做体积保护
                with requests.get(script_cfg, stream=True, timeout=_HTTP_TIMEOUT) as r:
                    r.raise_for_status()
                    total = 0
                    chunks = []
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # 过滤 keep-alive 块
                            total += len(chunk)
                            if total > _MAX_SCRIPT_BYTES:
                                raise ValueError(
                                    f"远端脚本过大，超过 {_MAX_SCRIPT_BYTES} 字节上限: {script_cfg}",
                                )
                            chunks.append(chunk)
                    # 尝试按响应声明的编码解码，否则回退到 utf-8
                    encoding = r.encoding or "utf-8"
                    job.script = b"".join(chunks).decode(encoding, errors="replace")
            except requests.RequestException as e:
                raise RuntimeError(f"拉取远端脚本失败: {script_cfg}，原因: {e}") from e
        else:
            # 其余情况按“本地路径或内联内容”处理：如果存在当文件读；否则视为脚本文本
            path = Path(script_cfg)
            if path.exists() and path.is_file():
                with open(path, encoding="utf-8") as f:
                    job.script = f.read()
            else:
                raise ValueError("无法理解的 script 内容")

    return job
