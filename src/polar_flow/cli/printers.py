# utils/console.py
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

_console = Console()


# 圆角面板
def _panel(
    msg: str | Text,
    *,
    title: str | None = None,
    style: str = "none",
    border: str = "cyan",
) -> Panel:
    return Panel(
        msg if isinstance(msg, Text) else Text(str(msg)),
        title=title,
        title_align="left",
        box=box.ROUNDED,  # 圆角
        border_style=border,  # 边框颜色
        expand=True,
        style=style,  # 面板内文字的基础样式
        padding=(1, 2),
    )


def print_info(msg: str, title: str = "INFO") -> None:
    _console.print(_panel(msg, title=title, border="cyan"))


def print_success(msg: str, title: str = "OK") -> None:
    _console.print(_panel(msg, title=title, border="green"))


def print_warning(msg: str, title: str = "WARN") -> None:
    _console.print(_panel(msg, title=title, border="yellow"))


def print_error(msg: str, title: str = "ERROR") -> None:
    _console.print(_panel(msg, title=title, border="red"))


def print_debug(msg: str, title: str = "DEBUG", debug: bool = False) -> None:
    if debug:
        _console.print(_panel(Text(str(msg), style="dim"), title=title, border="bright_black"))


# 打印键值对
def print_kv(title: str, mapping: dict[str, object], as_what: str) -> None:
    if as_what == "table":
        table = Table(title=title, safe_box=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta", overflow="fold")

        for k, v in mapping.items():
            table.add_row(str(k), str(v))

        _console.print(table)
    else:
        table = Table(title=title, show_lines=True, expand=True, box=None)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for k, v in mapping.items():
            table.add_row(str(k), Pretty(v, expand_all=True))

        _console.print(table)
