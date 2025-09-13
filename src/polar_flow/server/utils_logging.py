# server/utils_logging.py
from __future__ import annotations

import os
from pathlib import Path
import shlex

MAX_KEEP = 16 * 1024  # 16KB 片段

# 敏感信息
SENSITIVE_ENV_KEYS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASS",
    "KEY",
    "AWS_",
    "GCP_",
    "AZURE_",
    "OPENAI_",
    "HUGGINGFACE_",
    "HF_",
    "SLACK_",
    "GITHUB_",
    "GITLAB_",
    "DOCKERHUB_",
    "NPM_",
    "PYPI_",
)


def _snippet(s: str, keep: int = MAX_KEEP) -> str:
    if s is None:
        return ""
    if len(s) <= keep:
        return s
    head = s[: keep // 2]
    tail = s[-keep // 2 :]
    return head + "\n...[TRUNCATED]...\n" + tail


def save_task_logs(task, stdout: str, stderr: str) -> tuple[str, str, str, str]:
    wdir = Path(task.working_dir or os.getcwd())
    logdir = wdir / ".polar_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    out_path = logdir / f"task_{task.id}.out"
    err_path = logdir / f"task_{task.id}.err"
    out_path.write_text(stdout or "", encoding="utf-8", errors="ignore")
    err_path.write_text(stderr or "", encoding="utf-8", errors="ignore")
    return (
        out_path.as_posix(),
        err_path.as_posix(),
        _snippet(stdout or ""),
        _snippet(stderr or ""),
    )


def format_argv(argv: list[str]) -> str:
    """把 argv 美观地拼为一行 shell 命令（仅用于日志）。"""
    try:
        return shlex.join(argv)
    except Exception:
        # 兼容极端情况
        return " ".join(repr(x) for x in argv)


def redact_env(env: dict[str, str], keys_whitelist: set[str] | None = None) -> dict[str, str]:
    """
    返回一个适合写日志的 env 摘要：
    - 仅包含 whitelist 指定的键（若提供）
    - 对包含敏感关键词的键做脱敏（显示 '***'）
    """
    keys = set(env.keys())
    if keys_whitelist is not None:
        keys &= keys_whitelist
    out: dict[str, str] = {}
    for k in sorted(keys):
        v = env.get(k, "")
        upk = k.upper()
        if any(tok in upk for tok in SENSITIVE_ENV_KEYS):
            out[k] = "***"
        else:
            out[k] = v
    return out
