from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from rich.traceback import Traceback

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


def _print_http_error(action: str, e: requests.HTTPError, *, debug: bool) -> None:
    """统一友好错误输出；--debug 时打印调用栈"""
    resp = getattr(e, "response", None)
    detail = None
    if resp is not None:
        with contextlib.suppress(Exception):
            j = resp.json()
            detail = j.get("error") or j.get("message")
        if not detail:
            # 回退到纯文本
            with contextlib.suppress(Exception):
                detail = resp.text.strip()
    # 裁剪特别长的 HTML/文本，避免把整页 HTML 打到终端
    raw_msg = detail or str(e)
    MAX_LEN = 2000
    msg = (
        raw_msg
        if len(raw_msg) <= MAX_LEN
        else (raw_msg[:1000] + "\n\n\n...[TRUNCATED]...\n\n\n" + raw_msg[-800:])
    )
    console.print(pretty_panel(f"{action}", content=Text(msg, style="red")))
    if debug:
        # 打印带语法高亮的异常追踪
        tb = Traceback.from_exception(e.__class__, e, e.__traceback__, show_locals=False)
        console.print(tb)


def badge(text: str, style: str) -> Text:
    return Text(f" {text} ", style=style)


def fmt_status(s: str) -> Text:
    label, style = STATUS_STYLES.get(s.upper(), (s, "bold"))
    return badge(label, style)


def safe_get(d: dict, *keys: Any, default: Any = "") -> Any:
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
                vs.append(Text(str(cur)))
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
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
        )
        r.raise_for_status()
        self._save_cookies()
        try:
            return r.json()
        except Exception:  # noqa: BLE001
            return {"message": "login ok"}

    def logout(self) -> dict:
        r = self.session.post(f"{self.base_url}/auth/logout")
        r.raise_for_status()
        # 清空内存中的 cookies
        self.session.cookies.clear()
        try:
            if COOKIE_FILE.exists():
                COOKIE_FILE.unlink()
        except Exception:
            pass
        return r.json()

    # ---- Tasks ----
    def create_task(self, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks", json=payload)
        r.raise_for_status()
        return r.json()

    def check_task(self, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/api/tasks_check", json=payload)
        r.raise_for_status()
        return r.json()

    def list_tasks(self, status: str | None = None) -> list[dict]:
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
        r.raise_for_status()
        return r.json()

    def create_user(self, username: str, password: str) -> dict:
        r = self.session.post(
            f"{self.base_url}/api/admin/users",
            json={"username": username, "password": password},
        )
        r.raise_for_status()
        return r.json()

    def list_users(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/api/admin/users")
        r.raise_for_status()
        return r.json()

    def patch_user(self, user_id: int, payload: dict) -> dict:
        r = self.session.patch(f"{self.base_url}/api/admin/users/{user_id}", json=payload)
        r.raise_for_status()
        return r.json()

    def whoami(self) -> dict:
        r = self.session.get(f"{self.base_url}/me")
        r.raise_for_status()
        return r.json()

    def get_user(self, user_id: int) -> dict:
        r = self.session.get(f"{self.base_url}/api/admin/users/{user_id}")
        r.raise_for_status()
        return r.json()


# ---------------- CLI commands ----------------
@app.command()
def login(
    username: str = typer.Argument(None, help="用户名"),
    password: str = typer.Option(
        ...,
        "--password",
        "-p",
        prompt=True,
        hide_input=True,
        help="密码",
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """登录并保存会话 Cookie。"""
    c = Client(base_url)
    with console.status("[bold]登录中..."):
        try:
            res = c.login(username, password)
        except requests.HTTPError as e:
            _print_http_error("登录失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    user = safe_get(res, "user", "username", default=username)
    console.print(
        pretty_panel(
            "登陆成功",
            content=Text("登录为 ") + Text(user, style="bold cyan"),
        ),
    )
    console.print(Text.from_markup(f"[dim]Cookie 保存到 {COOKIE_FILE}[/]"))


@app.command()
def logout(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """注销当前会话。"""
    c = Client(base_url)
    with console.status("[bold]登出中..."):
        try:
            res = c.logout()
        except requests.HTTPError as e:
            _print_http_error("登出失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
    else:
        msg = res.get("message", "登出")
        console.print(pretty_panel("登出", content=Text(msg, style="green")))


@app.command("gpus")
def gpus_cmd(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """查看 GPU 状态。"""
    c = Client(base_url)
    with console.status("[bold]获取 GPU 状态中..."):
        try:
            infos = c.list_gpus()
        except requests.HTTPError as e:
            _print_http_error("获取 GPU 状态失败", e, debug=debug)
            raise typer.Exit(1)

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
                    cur = cur.get(part, "") if isinstance(cur, dict) else ""
                if key.lower().endswith("status"):
                    row.append(fmt_status(str(cur)))
                else:
                    row.append(Text(str(cur)))
            table.add_row(*row)
        console.print(pretty_panel("GPU 状态", content=table))
    else:
        # 回退：每块 GPU 显示 JSON
        for i, g in enumerate(infos, 1):
            console.print(pretty_panel(f"GPU #{i}", content=RICH_JSON.from_data(g)))


@app.command("submit")
def submit_cmd(
    config: str | None = typer.Option(None, "--config", "-c"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """从 TOML 提交任务。"""
    if config is None:
        data = toml.load(
            typer.Option(
                ...,
                "--config",
                "-c",
                exists=True,
                readable=True,
                help="TOML 任务配置文件",
            ),
        )
    else:
        data = toml.load(Path(config))
    t = data.get("task", {})
    payload = {
        "name": t.get("name"),
        "command": t.get("command"),
        "requested_gpus": t.get("requested_gpus", "AUTO:1"),
        "working_dir": t.get("working_dir", str(Path.cwd())),
        "gpu_memory_limit": t.get("gpu_memory_limit"),
        "priority": t.get("priority", 100),
        "docker_image": t.get("docker_image"),
        "docker_args": t.get("docker_args"),
        "env": t.get("env"),
    }
    c = Client(base_url)
    with console.status("[bold]提交任务中..."):
        try:
            res = c.create_task(payload)
        except requests.HTTPError as e:
            _print_http_error("提交任务失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    info = Table(box=box.SIMPLE, show_header=False)
    info.add_row("ID", str(res.get("id", "")))
    info.add_row("Name", str(res.get("name", "")))
    info.add_row("Status", fmt_status(str(res.get("status", ""))))
    info.add_row("Priority", str(res.get("priority", "")))
    console.print(pretty_panel("任务提交成功", content=info))


@app.command("check")
def check_cmd(
    config: str | None = typer.Option(None, "--config", "-c"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """从 TOML 检查任务是否可行。"""
    if config is None:
        data = toml.load(
            typer.Option(
                ...,
                "--config",
                "-c",
                exists=True,
                readable=True,
                help="TOML 任务配置文件",
            ),
        )
    else:
        data = toml.load(Path(config))
    t = data.get("task", {})
    payload = {
        "name": t.get("name"),
        "command": t.get("command"),
        "requested_gpus": t.get("requested_gpus", "AUTO:1"),
        "working_dir": t.get("working_dir", str(Path.cwd())),
        "gpu_memory_limit": t.get("gpu_memory_limit"),
        "priority": t.get("priority", 100),
        "docker_image": t.get("docker_image"),
        "docker_args": t.get("docker_args"),
        "env": t.get("env"),
    }
    c = Client(base_url)
    with console.status("[bold]检查任务中..."):
        try:
            res = c.check_task(payload)
        except requests.HTTPError as e:
            _print_http_error("检查任务失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    info = Table(box=box.SIMPLE, show_header=False)
    info.add_row("cmd", str(res.get("cmd", "")))
    console.print(pretty_panel("检查提交成功", content=info))


@app.command("ls")
def list_cmd(
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="过滤任务状态 (PENDING/RUNNING/SUCCESS/FAILED/CANCELLED)",
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """列出我的任务。"""
    c = Client(base_url)
    with console.status("[bold]Loading tasks..."):
        try:
            items = c.list_tasks(status=status)
        except requests.HTTPError as e:
            _print_http_error("列出任务失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(items))
        return

    if not items:
        console.print(pretty_panel("任务", content=Text("没有任务.", style="yellow")))
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
    subtitle = f"筛选: {status}" if status else None
    console.print(pretty_panel("我的任务", subtitle=subtitle, content=table))


@app.command("logs")
def logs_cmd(
    task_id: int = typer.Argument(...),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """查看任务日志（stdout/stderr）。"""
    c = Client(base_url)
    with console.status("[bold]获取日志中..."):
        try:
            t = c.get_task(task_id)
        except requests.HTTPError as e:
            _print_http_error("获取日志失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(
            RICH_JSON.from_data(
                {"stdout": t.get("stdout_log") or "", "stderr": t.get("stderr_log") or ""},
            ),
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
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """取消指定任务。"""
    c = Client(base_url)
    with console.status("[bold]取消任务中..."):
        try:
            res = c.cancel_task(task_id)
        except requests.HTTPError as e:
            _print_http_error("取消任务失败", e, debug=debug)
            raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(res))
        return

    msg = res.get("message", "ok")
    style = "green" if "ok" in msg.lower() or "success" in msg.lower() else "yellow"
    console.print(pretty_panel("取消任务", content=Text(msg, style=style)))


@app.command("users")
def users_cmd(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """列出所有用户（仅管理员）。"""
    c = Client(base_url)
    try:
        items = c.list_users()
    except requests.HTTPError as e:
        _print_http_error("列出所有用户失败", e, debug=debug)
        raise typer.Exit(1)
    for u in items:
        typer.echo(
            f"#{u['id']} {u['username']} role={u['role']} prio={u['priority']} visible={u['visible_gpus']}"
        )


@app.command("add-user")
def add_user_cmd(
    username: str = typer.Option(..., "--username", "-u"),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """新建用户（仅管理员）。"""
    c = Client(base_url)
    # 客户端侧的小校验：避免 400（密码最少 6 位）
    if len(password) < 6:
        console.print(
            pretty_panel(
                "创建用户失败",
                content=Text("密码长度应大于 6 位.", style="red"),
            ),
        )
        raise typer.Exit(2)
    try:
        u = c.create_user(username, password)
    except requests.HTTPError as e:
        _print_http_error("创建用户失败", e, debug=debug)
        raise typer.Exit(1)
    typer.echo(f"创建用户: {u['username']} (id={u['id']})")


@app.command("patch-user")
def patch_user_cmd(
    user_id: int = typer.Argument(..., help="用户ID"),
    role: str = typer.Option(None, "--role", "-r", help="user|admin"),
    priority: int = typer.Option(None, "--priority", help="优先级"),
    visible_gpus: str = typer.Option(None, "--visible-gpus", help="逗号分隔的 GPU 列表"),
    password: str = typer.Option(None, "--password", "-p"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
) -> None:
    """修改用户属性（仅管理员）。"""
    payload = {}
    if role:
        payload["role"] = role
    if priority is not None:
        payload["priority"] = priority
    if visible_gpus is not None:
        payload["visible_gpus"] = [int(x) for x in visible_gpus.split(",") if x.strip()]
    if password:
        payload["password"] = password
    c = Client(base_url)
    try:
        u = c.patch_user(user_id, payload)
    except requests.HTTPError as e:
        _print_http_error("修改用户属性失败", e, debug=debug)
        raise typer.Exit(1)
    typer.echo(f"User updated: {u['username']} (id={u['id']}) -> {u}")


@app.command("whoami")
def whoami_cmd(
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", "-b"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug 模式，打印调用栈"),
    json_out: bool = typer.Option(False, "--json", "-j", help="输出原始 JSON 数据"),
) -> None:
    """显示当前登录用户信息（需要登录）。"""
    c = Client(base_url)
    try:
        me = c.whoami()
    except requests.HTTPError as e:
        _print_http_error("WhoAmI", e, debug=debug)
        raise typer.Exit(1)

    if json_out:
        console.print(RICH_JSON.from_data(me))
        return

    info = Table(box=box.SIMPLE, show_header=False)
    info.add_row("ID", str(me.get("id", "")))
    info.add_row("Username", str(me.get("username", "")))
    info.add_row("Role", str(me.get("role", "")))
    info.add_row("Priority", str(me.get("priority", "")))
    info.add_row("Visible GPUs", ", ".join(map(str, me.get("visible_gpus") or [])))
    console.print(pretty_panel("当前用户", content=info))


def main() -> None:
    app()
