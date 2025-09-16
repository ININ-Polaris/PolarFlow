# utils/console.py
from __future__ import annotations
from collections.abc import Iterable

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
        table = Table(title=title, safe_box=True, expand=False)
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


def print_kv_grouped(
    title: str,
    groups: dict[str, dict[str, object]],
    as_what: str = "table",
    *,
    group_order: Iterable[str] | None = None,
    sort_keys: bool = True,
    empty_placeholder: str = "—",
) -> None:
    """
    分组打印键值对：
      - groups: { group_name: { key: value, ... }, ... }
      - as_what: "table" -> 值以 str 渲染；其他 -> Pretty(v, expand_all=True)
      - group_order: 指定分组顺序（可选），未列出的分组排后面
      - show_empty_groups: 是否打印空分组
    """
    # 基础表格风格与列
    if as_what == "table":
        tbl = Table(title=title, safe_box=True, expand=False)
        tbl.add_column("Group", style="bold yellow", no_wrap=True)
        tbl.add_column("Key", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="magenta", overflow="fold")
    else:
        tbl = Table(title=title, show_lines=True, expand=False, box=box.SIMPLE_HEAVY)
        tbl.add_column("Group", style="bold yellow", no_wrap=True)
        tbl.add_column("Key", style="bold cyan", no_wrap=True)
        tbl.add_column("Value", style="magenta")

    # 准备分组顺序
    all_group_names = list(groups.keys())
    ordered: list[str] = []
    if group_order:
        seen = set()
        for g in group_order:
            if g in all_group_names and g not in seen:
                ordered.append(g)
                seen.add(g)
        for g in all_group_names:
            if g not in seen:
                ordered.append(g)
    else:
        ordered = all_group_names[:]

    for g in ordered:
        inner = groups.get(g, {})

        # 组头行
        tbl.add_row(f"[bold]{g}[/bold]", "", "")

        # 组内键排序
        keys = list(inner.keys())
        if sort_keys:
            keys.sort()

        if keys:
            for k in keys:
                v = inner[k]
                rendered = str(v) if as_what == "table" else Pretty(v, expand_all=True)
                tbl.add_row("", str(k), rendered)
        else:
            # 空分组的占位
            tbl.add_row("", empty_placeholder, empty_placeholder)

        if g != ordered[-1]:
            tbl.add_section()

    _console.print(tbl)
