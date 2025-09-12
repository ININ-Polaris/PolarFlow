from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import colorama
import requests
import toml
import typer
from rich import box
from rich.console import Console
from rich.json import JSON as RICH_JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(add_completion=False, help="BIT ININ 课题组自用 服务器 GPU 资源分配器")

DEFAULT_BASE_URL = os.environ.get("POLAR_BASE_URL", "http://127.0.0.1:5000")
STATE_DIR = Path(os.environ.get("POLAR_STATE_DIR", "~/.polar_flow")).expanduser()
COOKIE_FILE = STATE_DIR / "cookies.txt"

console = Console(highlight=False)

STATUS_STYLES = {
    "PENDING": ("PENDING", "bold yellow"),
    "RUNNING": ("RUNNING", "bold cyan"),
    "SUCCESS": ("SUCCESS", "bold green"),
    "FAILED": ("FAILED", "bold red"),
    "CANCELLED": ("CANCELLED", "bold magenta"),
}


def badge(text: str, style: str) -> Text:
    return Text(f" {text} ", style=style)


def fmt_status(s: str) -> Text:
    label, style = STATUS_STYLES.get(s.upper(), (s, "bold"))
    return badge(label, style)


def safe_get(d: dict, *keys, default: Any = "") -> Any:
    for k in keys:
        d = d.get(k, {})
    return d or default


def to_table_from_dicts(rows: Iterable[dict], columns: list[tuple[str, str]]) -> Table:
    """columns: list of (header, key) where key supports dotted path like 'gpu.name'."""
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold")
    for header, _ in columns:
        table.add_column(header, overflow="fold", no_wrap=False)
    for r in rows:
        vs = []
        for _, key in columns:
            cur: Any = r
            for part in key.split("."):
                cur = cur.get(part, "") if isinstance(cur, dict) else ""
            if key.lower().endswith("status"):
                vs.append(fmt_status(str(cur)))
            else:
                vs.append(str(cur))
        table.add_row(*vs)
    return table


def pretty_panel(title: str, subtitle: str | None = None, content: Any = "") -> Panel:
    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
    )


class Client:
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if COOKIE_FILE.exists():
            with contextlib.suppress(Exception):
                self.session.cookies.update(
                    requests.utils.cookiejar_from_dict(json.loads(COOKIE_FILE.read_text())),
                )

    def _save_cookies(self) -> None:
        COOKIE_FILE.write_text(json.dumps(requests.utils.dict_from_cookiejar(self.session.cookies)))

    # ---- Auth ----
    def login(self, username: str, password: str) -> dict:
        r = self.session.post(
            f"{self.base_url}/auth/login", json={"username": username, "password": password}
        )
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                err = r.json().get("error")
            except Exception:  # noqa: BLE001
                err = r.text
            msg = f"{colorama.Fore.BLUE}[登录失败]: {colorama.Fore.RED}{err} ({e}){colorama.Style.RESET_ALL}"
        # 成功：保存 cookies 并返回体
        self._save_cookies()
        try:
            return r.json()
        except Exception:
            return {"message": "login ok"}

    def logout(self) -> dict:
        r = self.session.post(f"{self.base_url}/auth/logout")
        r.raise_for_status()
        self._save_cookies()
        return r.json()

    # ---- Tasks ----
    def create_task(self, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks", json=payload)
        r.raise_for_status()
        return r.json()

    def list_tasks(self, status: Optional[str] = None) -> list[dict]:
        params = {"status": status} if status else None
        r = self.session.get(f"{self.base_url}/api/tasks", params=params)
        r.raise_for_status()
        return r.json()

    def get_task(self, task_id: int) -> dict:
        r = self.session.get(f"{self.base_url}/api/tasks/{task_id}")
        r.raise_for_status()
        return r.json()

    def cancel_task(self, task_id: int) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks/{task_id}/cancel")
        r.raise_for_status()
        return r.json()

    def list_gpus(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/api/gpus")
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                err = r.json().get("error")
            except Exception:
                err = r.text
            raise SystemExit(
                f"{colorama.Fore.BLUE}[查询失败]: {colorama.Fore.RED}{err} ({e}){colorama.Style.RESET_ALL}"
            )
        return r.json()


# ---------------- CLI commands ----------------
@app.command()
def login(
    username: str = typer.Option(..., "--username", "-u"),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """登录并保存会话 Cookie。"""
    c = Client(base_url)
    with console.status("[bold]Logging in..."):
        res = c.login(username, password)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    user = safe_get(res, "user", "username", default=username)
    console.print(pretty_panel("Login Success", content=Text(f"Logged in as [bold cyan]{user}[/]")))
    console.print(f"[dim]Cookie saved to {COOKIE_FILE}[/]")


@app.command()
def logout(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """注销当前会话。"""
    c = Client(base_url)
    with console.status("[bold]Logging out..."):
        try:
            res = c.logout()
        except requests.HTTPError as e:
            console.print(pretty_panel("Logout Failed", content=Text(str(e), style="red")))
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
    else:
        msg = res.get("message", "logged out")
        console.print(pretty_panel("Logout", content=Text(msg, style="green")))


@app.command("gpus")
def gpus_cmd(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """查看 GPU 状态。"""
    c = Client(base_url)
    with console.status("[bold]Fetching GPU status..."):
        infos = c.list_gpus()

    if json_out:
        console.print(RICH_JSON.from_data(infos))
        return

    # 尝试识别常见字段；若没有则逐条以 JSON 展示
    candidate_fields = [
        ("ID", "id"),
        ("Model", "name"),
        ("UUID", "uuid"),
        ("Util%", "utilization"),
        ("Mem Used", "memory.used"),
        ("Mem Total", "memory.total"),
        ("Temp°C", "temperature"),
        ("Power W", "power_draw"),
        ("Status", "status"),
        ("Processes", "processes"),
    ]

    has_any = any(isinstance(g, dict) and any(k in g for _, k in candidate_fields) for g in infos)
    if has_any:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold", show_lines=False)
        for h, _ in candidate_fields:
            table.add_column(h, overflow="fold", justify="center")
        for g in infos:
            row = []
            for _, key in candidate_fields:
                cur: Any = g
                for part in key.split("."):
                    if isinstance(cur, dict):
                        cur = cur.get(part, "")
                    else:
                        cur = ""
                if key.lower().endswith("status"):
                    row.append(fmt_status(str(cur)))
                else:
                    row.append(str(cur))
            table.add_row(*row)
        console.print(pretty_panel("GPU Status", content=table))
    else:
        # 回退：每块 GPU 显示 JSON
        for i, g in enumerate(infos, 1):
            console.print(pretty_panel(f"GPU #{i}", content=RICH_JSON.from_data(g)))


@app.command("submit")
def submit_cmd(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True, help="TOML 任务配置文件"
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """从 TOML 提交任务。"""
    data = toml.load(config)
    t = data.get("task", {})
    payload = {
        "name": t.get("name"),
        "command": t.get("command"),
        "requested_gpus": t.get("requested_gpus", "AUTO:1"),
        "working_dir": t.get("working_dir", str(Path.cwd())),
        "gpu_memory_limit": t.get("gpu_memory_limit"),
        "priority": t.get("priority", 100),
    }
    c = Client(base_url)
    with console.status("[bold]Submitting task..."):
        res = c.create_task(payload)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    info = Table(box=box.SIMPLE, show_header=False)
    info.add_row("ID", str(res.get("id", "")))
    info.add_row("Name", str(res.get("name", "")))
    info.add_row("Status", fmt_status(str(res.get("status", ""))))
    info.add_row("Priority", str(res.get("priority", "")))
    console.print(pretty_panel("Task Submitted", content=info))


@app.command("ls")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", help="过滤任务状态 (PENDING/RUNNING/SUCCESS/FAILED/CANCELLED)"
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """列出我的任务。"""
    c = Client(base_url)
    with console.status("[bold]Loading tasks..."):
        items = c.list_tasks(status=status)

    if json_out:
        console.print(RICH_JSON.from_data(items))
        return

    if not items:
        console.print(pretty_panel("Tasks", content=Text("No tasks found.", style="yellow")))
        return

    cols = [
        ("ID", "id"),
        ("Status", "status"),
        ("Name", "name"),
        ("Prio", "priority"),
        ("Created At", "created_at"),
    ]
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold", show_lines=False)
    for h, _ in cols:
        table.add_column(h, overflow="fold", justify="center" if h in {"ID", "Prio"} else "left")
    for it in items:
        table.add_row(
            str(it.get("id", "")),
            fmt_status(str(it.get("status", ""))),
            str(it.get("name", "")),
            str(it.get("priority", "")),
            str(it.get("created_at", "")),
        )
    subtitle = f"Filter: {status}" if status else None
    console.print(pretty_panel("My Tasks", subtitle=subtitle, content=table))


@app.command("logs")
def logs_cmd(
    task_id: int = typer.Argument(...),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """查看任务日志（stdout/stderr）。"""
    c = Client(base_url)
    with console.status("[bold]Fetching logs..."):
        t = c.get_task(task_id)

    if json_out:
        console.print(
            RICH_JSON.from_data(
                {"stdout": t.get("stdout_log") or "", "stderr": t.get("stderr_log") or ""}
            )
        )
        return

    stdout = t.get("stdout_log") or ""
    stderr = t.get("stderr_log") or ""

    out_syntax = Syntax(stdout, "bash", word_wrap=True, line_numbers=False)
    err_syntax = Syntax(stderr, "bash", word_wrap=True, line_numbers=False)

    console.print(pretty_panel("STDOUT", content=out_syntax))
    console.print(pretty_panel("STDERR", content=err_syntax))


@app.command("cancel")
def cancel_cmd(
    task_id: int = typer.Argument(...),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """取消指定任务。"""
    c = Client(base_url)
    with console.status("[bold]Cancelling task..."):
        res = c.cancel_task(task_id)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    msg = res.get("message", "ok")
    style = "green" if "ok" in msg.lower() or "success" in msg.lower() else "yellow"
    console.print(pretty_panel("Cancel Task", content=Text(msg, style=style)))


def main() -> None:
    app()
