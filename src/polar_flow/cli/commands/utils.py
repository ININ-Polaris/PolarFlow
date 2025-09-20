from __future__ import annotations

import re
import tomllib
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


# toml_to_slurm_sh
def toml_to_slurm_sh(
    toml_str: str,
    job_desc_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    将 TOML 作业描述转换为 v0.0.43 job_desc_msg 风格的字典（顶级包含 script 与各作业字段）。
    变更点：
      1) 返回 dict（非 JSON 字符串）
      2) 新增 job_desc_override，且其键值具有最高优先级（最终覆盖）
      3) 顶级扁平：{"script": "...", "name": "...", "partition": "...", ...}

    约定映射（尽量贴近 v0.0.43_job_desc_msg 的字段）：
      - time -> time_limit（单位：分钟）
      - mem -> memory_per_node（单位：MiB）
      - mem_per_cpu -> memory_per_cpu（MiB）
      - constraint -> constraints（逗号串）
      - chdir -> work_dir（若站点字段名不同，可用 override 覆盖）
      - output/error -> standard_output/standard_error
      - gpus -> tres_per_job="gpu:N"（并保留 gres="gpu:N" 以兼容）
      - array / dependency 支持列表或字符串（拼接为逗号串）
      - env.exports -> environment=[{"name": K, "value": V}, ...]

    提醒：不同集群的 OpenAPI 字段名可能略有差异，请用 job_desc_override 精确覆盖。
    """

    # ----------------- helpers -----------------
    def _ensure_dict(obj: Any, path: str) -> dict:
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise TypeError(f"{path} 必须是表/对象（TOML table），实际收到: {type(obj)}")
        return obj

    def _norm_str_or_list(x: Any, path: str) -> list[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple)):
            out: list[str] = []
            for i, item in enumerate(x):
                if not isinstance(item, str):
                    raise TypeError(f"{path}[{i}] 必须是字符串，实际收到: {type(item)}")
                out.append(item)
            return out
        raise TypeError(f"{path} 必须是字符串或字符串列表，实际收到: {type(x)}")

    def _parse_time_to_minutes(t: Any) -> int | None:
        """支持 'DD-HH:MM:SS' / 'HH:MM:SS' / 'MM:SS' / 'H:MM' 字符串；数值视为分钟。返回整数分钟。"""
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return int(t)
        if isinstance(t, str):
            s = t.strip()
            # DD-HH:MM:SS
            m = re.fullmatch(r"(\d+)-(\d{1,2}):(\d{2}):(\d{2})", s)
            if m:
                dd, hh, mm, ss = map(int, m.groups())
                return dd * 24 * 60 + hh * 60 + mm + (1 if ss > 0 else 0)
            # HH:MM:SS
            m = re.fullmatch(r"(\d{1,3}):(\d{2}):(\d{2})", s)
            if m:
                hh, mm, ss = map(int, m.groups())
                return hh * 60 + mm + (1 if ss > 0 else 0)
            # MM:SS
            m = re.fullmatch(r"(\d{1,4}):(\d{2})", s)
            if m:
                mm, ss = map(int, m.groups())
                return mm + (1 if ss > 0 else 0)
            # 其他字符串（如 "infinite"）不转分钟：由 override 指定或让集群侧解析
            return None
        raise TypeError(f"job.time 必须是字符串或数值（分钟），实际收到: {type(t)}")

    def _mem_to_mib(mem: Any) -> int | None:
        """将 '64G' / '64000M' / 数字 等转换为 MiB（整数）。"""
        if mem is None:
            return None
        if isinstance(mem, (int, float)):
            return int(mem)
        if isinstance(mem, str):
            s = mem.strip().upper()
            m = re.fullmatch(r"(\d+)([KMGTP]?)(I?B)?", s)
            if not m:
                raise ValueError(f"无法解析内存规格: {mem}")
            num = int(m.group(1))
            unit = m.group(2)
            if unit in {"", "M"}:
                return num
            if unit == "K":
                return max(1, num // 1024)
            if unit == "G":
                return num * 1024
            if unit == "T":
                return num * 1024 * 1024
            if unit == "P":
                return num * 1024 * 1024 * 1024
        raise TypeError(f"内存必须是数字或字符串，收到: {type(mem)}")

    # ----------------- load & split -----------------
    cfg = tomllib.loads(toml_str)
    job = _ensure_dict(cfg.get("job"), "job")
    env = _ensure_dict(cfg.get("env"), "env")
    run = _ensure_dict(cfg.get("run"), "run")

    # ----------------- build top-level job_desc_msg-like dict -----------------
    out: dict[str, Any] = {}

    # 基础映射
    direct_map = {
        "name": "name",
        "partition": "partition",
        "qos": "qos",
        "account": "account",
        "ntasks": "ntasks",
        "ntasks_per_node": "ntasks_per_node",
        "cpus_per_task": "cpus_per_task",
        "reservation": "reservation",
        "nodelist": "required_nodes",  # 更通用字段名（逗号分隔）
        "exclude": "excluded_nodes",
        "begin": "begin_time",  # 可用 "now+1hour" 或时间串/UNIX 时间（见 schema 释义）
        "requeue": "requeue",
        "license": "licenses",
        "profile": "profile",  # 数组/枚举见 schema
        "hint": "hint",
        "threads_per_core": "threads_per_core",
    }
    for k, dst in direct_map.items():
        if k in job and job[k] is not None:
            out[dst] = job[k]

    # 约束/偏好
    if job.get("constraint"):
        out["constraints"] = (
            ",".join(_norm_str_or_list(job.get("constraint"), "job.constraint"))
            if isinstance(job.get("constraint"), (list, tuple))
            else str(job["constraint"])
        )

    # 时间：转为 time_limit（分钟）
    tl = _parse_time_to_minutes(job.get("time"))
    if tl is not None:
        out["time_limit"] = (
            tl  # v0.0.43 将 time_limit 定义为“最大运行时间（分钟）”:contentReference[oaicite:2]{index=2}
        )

    # 节点/内存
    if "nodes" in job:
        out["nodes"] = job["nodes"]
    if job.get("mem_per_cpu") is not None:
        out["memory_per_cpu"] = _mem_to_mib(
            job["mem_per_cpu"],
        )  # MB/MiB（schema 用“megabytes”描述）
    if job.get("mem") is not None:
        out["memory_per_node"] = _mem_to_mib(
            job["mem"],
        )  # MB/MiB（schema 用“megabytes”描述）

    # GPU：优先显式 gres，否则从 gpus 派生
    if job.get("gres"):
        out["gres"] = job["gres"]
    elif job.get("gpus") not in (None, 0, "0"):
        number = job["gpus"] if isinstance(job["gpus"], int) else str(job["gpus"])
        out["tres_per_job"] = f"gpu:{number}"
        out.setdefault("gres", f"gpu:{number}")

    # 数组/依赖
    dep = job.get("dependency")
    if isinstance(dep, (list, tuple)):
        dep = ",".join(str(x) for x in dep)
    if dep:
        out["dependency"] = dep
    arr = job.get("array")
    if isinstance(arr, (list, tuple)):
        arr = ",".join(str(x) for x in arr)
    if arr:
        out["array"] = arr

    # I/O 与工作目录
    if job.get("output"):
        out["standard_output"] = job["output"]
    if job.get("error"):
        out["standard_error"] = job["error"]
    if job.get("chdir"):
        out["current_working_directory"] = job["chdir"]

    # 独占
    if "exclusive" in job:
        out["exclusive"] = bool(job["exclusive"])

    # 环境变量 -> environment: [{"name": "...","value": "..."}]
    exports = env.get("exports") or {}
    if not isinstance(exports, dict):
        raise TypeError(f"env.exports 必须是表/对象，实际收到: {type(exports)}")
    if exports:
        out["environment"] = [{"name": str(k), "value": str(v)} for k, v in exports.items()]

    # ---------- 裸脚本（不包含任何 #SBATCH） ----------
    shell = run.get("shell") or "/bin/bash"
    if not isinstance(shell, str):
        raise TypeError(f"run.shell 必须是字符串，实际收到: {type(shell)}")
    lines: list[str] = [f"#!{shell}", ""]
    for line in _norm_str_or_list(run.get("pre"), "run.pre"):
        lines.append(line)
    if "command" in run:
        lines.extend(_norm_str_or_list(run.get("command"), "run.command"))
    if "commands" in run:
        lines.extend(_norm_str_or_list(run.get("commands"), "run.commands"))
    if len(lines) <= 2:
        lines.append("echo 'No run.command / run.commands specified'; exit 1")
    out["script"] = "\n".join(lines).rstrip() + "\n"

    # ---------- 额外 sbatch 选项（尽量归一；其余交给 override 或集群侧） ----------
    extra = cfg.get("extra_sbatch") or job.get("extra_sbatch")
    for raw in _norm_str_or_list(extra, "extra_sbatch"):
        s = raw.strip().removeprefix("#SBATCH").strip()
        if not s:
            continue
        if s.startswith("--profile=") and "profile" not in out:
            out["profile"] = _norm_str_or_list(s.split("=", 1)[1], "job.profile")
        elif s.startswith("--hint=") and "hint" not in out:
            out["hint"] = s.split("=", 1)[1]
        else:
            out.setdefault("generic_options", []).append(s)

    # ---------- 覆盖层（最高优先级） ----------
    if job_desc_override:
        if not isinstance(job_desc_override, dict):
            raise TypeError("job_desc_override 必须是字典")

        out = _merge(out, job_desc_override)

    return out


def merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """
    list/标量覆盖
    dict 递归合并
    dst <= src
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst
