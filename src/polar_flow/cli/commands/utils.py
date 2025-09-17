from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

Path = tuple[str | int, ...]
Predicate = Callable[[Any, Path], bool]
Replacer = Callable[[Any, Path], Any]


class UnhashableElementError(TypeError):
    pass


def replace_nested(
    obj: Any,
    predicate: Predicate,
    replacer: Replacer,
    *,
    in_place: bool = False,
    max_depth: int | None = None,
    visit_collections: bool = True,  # 是否也对容器自身做 predicate 检查
    set_unhashable: str = "stringify",  # "stringify" | "error" | "skip"
) -> Any:
    """
    通用嵌套替换：
      - 对任何位置的元素，如果 predicate(element, path) 为 True，则用 replacer(element, path) 的返回值替换
      - 递归进入 dict/list/tuple/set
      - 支持原地修改（dict/list），其他容器返回新对象
      - 传入 path（如 ("a", 0, "b")）方便在规则里判断位置
      - 防御循环引用

    obj : 任意可嵌套对象
    predicate : (value, path) -> bool
    replacer  : (value, path) -> new_value
    in_place  : dict/list 是否就地修改；tuple/set 仍返回新对象
    max_depth : 最深递归层数；None 表示不限制（根为深度 0）
    visit_collections : 为 True 时，容器对象本身也会先做一次 predicate 检查
    set_unhashable : 当 set 元素替换后不可哈希时的策略：
                     - "stringify": 转为 str 放入 set
                     - "skip"     : 跳过该元素
                     - "error"    : 抛出 UnhashableElementError
    """
    seen: set[int] = set()

    def _stringify_if_needed(x: Any) -> Any:
        try:
            hash(x)
        except TypeError:
            if set_unhashable == "stringify":
                return str(x)
            if set_unhashable == "skip":
                return None
            raise UnhashableElementError(f"Unhashable set element: {x!r}")  # noqa: B904
        else:
            return x

    def _walk(x: Any, path: Path, depth: int) -> Any:
        # 深度限制
        if max_depth is not None and depth > max_depth:
            return x

        # 先判断容器自身是否需要替换
        if visit_collections and predicate(x, path):
            return replacer(x, path)

        # 基本类型或已访问（循环引用）
        xid = id(x)
        if isinstance(x, (dict, list)) and xid in seen:
            return x
        # 对可变容器做循环检测
        if isinstance(x, (dict, list)):
            seen.add(xid)

        # dict
        if isinstance(x, dict):
            target = x if in_place else dict(x)
            for k in list(target.keys()):
                v = target[k]
                new_v = _walk(v, (*path, k), depth + 1)
                # 替换键值（不处理键本身，以免破坏映射）
                target[k] = new_v
            return target

        # list
        if isinstance(x, list):
            if in_place:
                for i, v in enumerate(x):
                    x[i] = _walk(v, (*path, i), depth + 1)
                return x
            return [_walk(v, (*path, i), depth + 1) for i, v in enumerate(x)]

        # tuple
        if isinstance(x, tuple):
            return tuple(_walk(v, (*path, i), depth + 1) for i, v in enumerate(x))

        # set
        if isinstance(x, set):
            new_set = set()
            for idx, v in enumerate(x):
                nv = _walk(v, (*path, f"<set:{idx}>"), depth + 1)
                nv2 = _stringify_if_needed(nv)
                if nv2 is None:  # skip
                    continue
                new_set.add(nv2)
            return new_set

        # 其他标量
        if predicate(x, path):
            return replacer(x, path)
        return x

    return _walk(obj, (), 0)


def no_val_nested(data: Any) -> Any:
    def _is_no_val_type(d: Any, _: Path) -> bool:
        if not isinstance(d, Mapping):
            return False
        return (
            {"set", "infinite", "number"}.issubset(d.keys())
            and isinstance(d.get("set"), bool)
            and isinstance(d.get("infinite"), bool)
            and isinstance(d.get("number"), (int, float))
        )

    def _to_value_or_inf(x: dict, _: Path) -> Any:
        return x["number"] if x["set"] else "INF"

    return replace_nested(
        data,
        _is_no_val_type,
        _to_value_or_inf,
        in_place=False,
    )


def version_nested(data: Any) -> Any:
    def _is(d: Any, _: Path) -> bool:
        if not isinstance(d, Mapping):
            return False
        return {"major", "micro", "minor"}.issubset(d.keys())

    def _to(x: dict, _: Path) -> Any:
        return f"{x['major']}.{x['micro']}.{x['minor']}"

    return replace_nested(
        data,
        _is,
        _to,
        in_place=False,
    )
