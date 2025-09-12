# server/utils_logging.py
from __future__ import annotations

import os
from pathlib import Path

MAX_KEEP = 16 * 1024  # 16KB 片段


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
