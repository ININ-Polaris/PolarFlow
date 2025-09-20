# generate_v0043_cli.py
# Purpose: Parse the provided attrs class for V0043JobDescMsg and generate a Typer CLI (v0043_cli.py)
# Notes:
# - Works off the embedded CLASS_SRC string below (paste your latest class text if it changes).
# - Translates inline triple-quoted comments to Chinese using a phrase dictionary; falls back to English.

import re
from pathlib import Path
from textwrap import indent

# ======= Paste/keep your class source here =======
CLASS_SRC = r'''
@_attrs_define
class V0043JobDescMsg:
    account: Union[Unset, str] = UNSET
    """ Account associated with the job """
    account_gather_frequency: Union[Unset, str] = UNSET
    """ Job accounting and profiling sampling intervals in seconds """
    admin_comment: Union[Unset, str] = UNSET
    """ Arbitrary comment made by administrator """
    allocation_node_list: Union[Unset, str] = UNSET
    """ Local node making the resource allocation """
    allocation_node_port: Union[Unset, int] = UNSET
    """ Port to send allocation confirmation to """
    argv: Union[Unset, list[str]] = UNSET
    array: Union[Unset, str] = UNSET
    """ Job array index value specification """
    batch_features: Union[Unset, str] = UNSET
    """ Features required for batch script's node """
    begin_time: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    flags: Union[Unset, list[V0043JobDescMsgFlagsItem]] = UNSET
    """ Job flags """
    burst_buffer: Union[Unset, str] = UNSET
    """ Burst buffer specifications """
    clusters: Union[Unset, str] = UNSET
    """ Clusters that a federated job can run on """
    cluster_constraint: Union[Unset, str] = UNSET
    """ Required features that a federated cluster must have to have a sibling job submitted to it """
    comment: Union[Unset, str] = UNSET
    """ Arbitrary comment made by user """
    contiguous: Union[Unset, bool] = UNSET
    """ True if job requires contiguous nodes """
    container: Union[Unset, str] = UNSET
    """ Absolute path to OCI container bundle """
    container_id: Union[Unset, str] = UNSET
    """ OCI container ID """
    core_specification: Union[Unset, int] = UNSET
    """ Specialized core count """
    thread_specification: Union[Unset, int] = UNSET
    """ Specialized thread count """
    cpu_binding: Union[Unset, str] = UNSET
    """ Method for binding tasks to allocated CPUs """
    cpu_binding_flags: Union[Unset, list[V0043JobDescMsgCpuBindingFlagsItem]] = UNSET
    """ Flags for CPU binding """
    cpu_frequency: Union[Unset, str] = UNSET
    """ Requested CPU frequency range <p1>[-p2][:p3] """
    cpus_per_tres: Union[Unset, str] = UNSET
    """ Semicolon delimited list of TRES=# values values indicating how many CPUs should be allocated for each
    specified TRES (currently only used for gres/gpu) """
    crontab: Union[Unset, "V0043CronEntry"] = UNSET
    deadline: Union[Unset, int] = UNSET
    """ Latest time that the job may start (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm
    (e.g., '[MM/DD[/YY]-]HH:MM[:SS]')) """
    delay_boot: Union[Unset, int] = UNSET
    """ Number of seconds after job eligible start that nodes will be rebooted to satisfy feature specification """
    dependency: Union[Unset, str] = UNSET
    """ Other jobs that must meet certain criteria before this job can start """
    end_time: Union[Unset, int] = UNSET
    """ Expected end time (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g.,
    '[MM/DD[/YY]-]HH:MM[:SS]')) """
    environment: Union[Unset, list[str]] = UNSET
    rlimits: Union[Unset, "V0043JobDescMsgRlimits"] = UNSET
    excluded_nodes: Union[Unset, list[str]] = UNSET
    extra: Union[Unset, str] = UNSET
    """ Arbitrary string used for node filtering if extra constraints are enabled """
    constraints: Union[Unset, str] = UNSET
    """ Comma-separated list of features that are required """
    group_id: Union[Unset, str] = UNSET
    """ Group ID of the user that owns the job """
    hetjob_group: Union[Unset, int] = UNSET
    """ Unique sequence number applied to this component of the heterogeneous job """
    immediate: Union[Unset, bool] = UNSET
    """ If true, exit if resources are not available within the time period specified """
    job_id: Union[Unset, int] = UNSET
    """ Job ID """
    kill_on_node_fail: Union[Unset, bool] = UNSET
    """ If true, kill job on node failure """
    licenses: Union[Unset, str] = UNSET
    """ License(s) required by the job """
    mail_type: Union[Unset, list[V0043JobDescMsgMailTypeItem]] = UNSET
    """ Mail event type(s) """
    mail_user: Union[Unset, str] = UNSET
    """ User to receive email notifications """
    mcs_label: Union[Unset, str] = UNSET
    """ Multi-Category Security label on the job """
    memory_binding: Union[Unset, str] = UNSET
    """ Binding map for map/mask_cpu """
    memory_binding_type: Union[Unset, list[V0043JobDescMsgMemoryBindingTypeItem]] = UNSET
    """ Method for binding tasks to memory """
    memory_per_tres: Union[Unset, str] = UNSET
    """ Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for
    each specified TRES (currently only used for gres/gpu) """
    name: Union[Unset, str] = UNSET
    """ Job name """
    network: Union[Unset, str] = UNSET
    """ Network specs for job step """
    nice: Union[Unset, int] = UNSET
    """ Requested job priority change """
    tasks: Union[Unset, int] = UNSET
    """ Number of tasks """
    oom_kill_step: Union[Unset, int] = UNSET
    """ Kill whole step in case of OOM in one of the tasks """
    open_mode: Union[Unset, list[V0043JobDescMsgOpenModeItem]] = UNSET
    """ Open mode used for stdout and stderr files """
    reserve_ports: Union[Unset, int] = UNSET
    """ Port to send various notification msg to """
    overcommit: Union[Unset, bool] = UNSET
    """ Overcommit resources """
    partition: Union[Unset, str] = UNSET
    """ Partition assigned to the job """
    distribution_plane_size: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    power_flags: Union[Unset, list[Any]] = UNSET
    prefer: Union[Unset, str] = UNSET
    """ Comma-separated list of features that are preferred but not required """
    hold: Union[Unset, bool] = UNSET
    """ Hold (true) or release (false) job (Job held) """
    priority: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    profile: Union[Unset, list[V0043JobDescMsgProfileItem]] = UNSET
    """ Profile used by the acct_gather_profile plugin """
    qos: Union[Unset, str] = UNSET
    """ Quality of Service assigned to the job """
    reboot: Union[Unset, bool] = UNSET
    """ Node reboot requested before start """
    required_nodes: Union[Unset, list[str]] = UNSET
    requeue: Union[Unset, bool] = UNSET
    """ Determines whether the job may be requeued """
    reservation: Union[Unset, str] = UNSET
    """ Name of reservation to use """
    script: Union[Unset, str] = UNSET
    """ Job batch script; only the first component in a HetJob is populated or honored """
    shared: Union[Unset, list[V0043JobDescMsgSharedItem]] = UNSET
    """ How the job can share resources with other jobs, if at all """
    site_factor: Union[Unset, int] = UNSET
    """ Site-specific priority factor """
    spank_environment: Union[Unset, list[str]] = UNSET
    distribution: Union[Unset, str] = UNSET
    """ Layout """
    time_limit: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    time_minimum: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    tres_bind: Union[Unset, str] = UNSET
    """ Task to TRES binding directives """
    tres_freq: Union[Unset, str] = UNSET
    """ TRES frequency directives """
    tres_per_job: Union[Unset, str] = UNSET
    """ Comma-separated list of TRES=# values to be allocated for every job """
    tres_per_node: Union[Unset, str] = UNSET
    """ Comma-separated list of TRES=# values to be allocated for every node """
    tres_per_socket: Union[Unset, str] = UNSET
    """ Comma-separated list of TRES=# values to be allocated for every socket """
    tres_per_task: Union[Unset, str] = UNSET
    """ Comma-separated list of TRES=# values to be allocated for every task """
    user_id: Union[Unset, str] = UNSET
    """ User ID that owns the job """
    wait_all_nodes: Union[Unset, bool] = UNSET
    """ If true, wait to start until after all nodes have booted """
    kill_warning_flags: Union[Unset, list[V0043JobDescMsgKillWarningFlagsItem]] = UNSET
    """ Flags related to job signals """
    kill_warning_signal: Union[Unset, str] = UNSET
    """ Signal to send when approaching end time (e.g. "10" or "USR1") """
    kill_warning_delay: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    current_working_directory: Union[Unset, str] = UNSET
    """ Working directory to use for the job """
    cpus_per_task: Union[Unset, int] = UNSET
    """ Number of CPUs required by each task """
    minimum_cpus: Union[Unset, int] = UNSET
    """ Minimum number of CPUs required """
    maximum_cpus: Union[Unset, int] = UNSET
    """ Maximum number of CPUs required """
    nodes: Union[Unset, str] = UNSET
    """ Node count range specification (e.g. 1-15:4) """
    minimum_nodes: Union[Unset, int] = UNSET
    """ Minimum node count """
    maximum_nodes: Union[Unset, int] = UNSET
    """ Maximum node count """
    minimum_boards_per_node: Union[Unset, int] = UNSET
    """ Boards per node required """
    minimum_sockets_per_board: Union[Unset, int] = UNSET
    """ Sockets per board required """
    sockets_per_node: Union[Unset, int] = UNSET
    """ Sockets per node required """
    threads_per_core: Union[Unset, int] = UNSET
    """ Threads per core required """
    tasks_per_node: Union[Unset, int] = UNSET
    """ Number of tasks to invoke on each node """
    tasks_per_socket: Union[Unset, int] = UNSET
    """ Number of tasks to invoke on each socket """
    tasks_per_core: Union[Unset, int] = UNSET
    """ Number of tasks to invoke on each core """
    tasks_per_board: Union[Unset, int] = UNSET
    """ Number of tasks to invoke on each board """
    ntasks_per_tres: Union[Unset, int] = UNSET
    """ Number of tasks that can access each GPU """
    minimum_cpus_per_node: Union[Unset, int] = UNSET
    """ Minimum number of CPUs per node """
    memory_per_cpu: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    memory_per_node: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    temporary_disk_per_node: Union[Unset, int] = UNSET
    """ Minimum tmp disk space required per node """
    selinux_context: Union[Unset, str] = UNSET
    """ SELinux context """
    required_switches: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    segment_size: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    standard_error: Union[Unset, str] = UNSET
    """ Path to stderr file """
    standard_input: Union[Unset, str] = UNSET
    """ Path to stdin file """
    standard_output: Union[Unset, str] = UNSET
    """ Path to stdout file """
    wait_for_switch: Union[Unset, int] = UNSET
    """ Maximum time to wait for switches in seconds """
    wckey: Union[Unset, str] = UNSET
    """ Workload characterization key """
    x11: Union[Unset, list[V0043JobDescMsgX11Item]] = UNSET
    """ X11 forwarding options """
    x11_magic_cookie: Union[Unset, str] = UNSET
    """ Magic cookie for X11 forwarding """
    x11_target_host: Union[Unset, str] = UNSET
    """ Hostname or UNIX socket if x11_target_port=0 """
    x11_target_port: Union[Unset, int] = UNSET
    """ TCP port """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)
'''

# ======= Simple EN->CN phrase dictionary (extend as needed) =======
CN_MAP = {
    "Account associated with the job": "与作业关联的账户",
    "Job accounting and profiling sampling intervals in seconds": "作业计费与性能采样的间隔（秒）",
    "Arbitrary comment made by administrator": "管理员填写的任意备注",
    "Local node making the resource allocation": "执行资源分配的本地节点",
    "Port to send allocation confirmation to": "用于发送分配确认的端口",
    "Job array index value specification": "作业数组索引值规范",
    "Features required for batch script's node": "批处理脚本节点所需的特性",
    "Job flags": "作业标志",
    "Burst buffer specifications": "突发缓冲区配置",
    "Clusters that a federated job can run on": "联合作业可运行的集群",
    "Required features that a federated cluster must have to have a sibling job submitted to it": "联合集群接受同级作业所需具备的特性",
    "Arbitrary comment made by user": "用户填写的任意备注",
    "True if job requires contiguous nodes": "若需要连续节点则为 True",
    "Absolute path to OCI container bundle": "OCI 容器包的绝对路径",
    "OCI container ID": "OCI 容器 ID",
    "Specialized core count": "专用核心数量",
    "Specialized thread count": "专用线程数量",
    "Method for binding tasks to allocated CPUs": "将任务绑定到已分配 CPU 的方法",
    "Flags for CPU binding": "CPU 绑定相关标志",
    "Requested CPU frequency range <p1>[-p2][:p3]": "请求的 CPU 频率范围 <p1>[-p2][:p3]",
    "Semicolon delimited list of TRES=# values values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)": "以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的 CPU 数量（目前仅用于 gres/gpu）",
    "Latest time that the job may start (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))": "作业最晚可开始的时间（UNIX 时间戳或 Slurm 识别的时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）",
    "Number of seconds after job eligible start that nodes will be rebooted to satisfy feature specification": "在作业满足可开始条件后，为满足特性要求而延迟重启节点的秒数",
    "Other jobs that must meet certain criteria before this job can start": "在本作业开始前必须满足条件的其他作业",
    "Expected end time (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))": "预期结束时间（UNIX 时间戳或 Slurm 识别的时间字符串，如 '[MM/DD[/YY]-]HH:MM[:SS]'）",
    "Arbitrary string used for node filtering if extra constraints are enabled": "启用额外约束时用于节点筛选的任意字符串",
    "Comma-separated list of features that are required": "必需特性的逗号分隔列表",
    "Group ID of the user that owns the job": "作业所属用户的组 ID",
    "Unique sequence number applied to this component of the heterogeneous job": "应用于此异构作业组件的唯一序号",
    "If true, exit if resources are not available within the time period specified": "若为 True，则在指定时间内资源不可用时退出",
    "Job ID": "作业 ID",
    "If true, kill job on node failure": "若为 True，当节点故障时杀死作业",
    "License(s) required by the job": "作业所需的许可证",
    "Mail event type(s)": "邮件事件类型",
    "User to receive email notifications": "接收邮件通知的用户",
    "Multi-Category Security label on the job": "作业上的多类别安全（MCS）标签",
    "Binding map for map/mask_cpu": "用于 map/mask_cpu 的绑定映射",
    "Method for binding tasks to memory": "任务与内存的绑定方法",
    "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)": "以分号分隔的 TRES=# 列表，表示每个指定 TRES 分配的内存（MB）（目前仅用于 gres/gpu）",
    "Job name": "作业名称",
    "Network specs for job step": "作业步骤的网络规格",
    "Requested job priority change": "请求的作业优先级变更",
    "Number of tasks": "任务数量",
    "Kill whole step in case of OOM in one of the tasks": "若某任务发生 OOM 则终止整个步骤",
    "Open mode used for stdout and stderr files": "stdout/stderr 文件的打开模式",
    "Port to send various notification msg to": "发送各类通知消息的端口",
    "Overcommit resources": "超量分配资源",
    "Partition assigned to the job": "作业所属分区",
    "Comma-separated list of features that are preferred but not required": "偏好但非必需的特性（逗号分隔）",
    "Profile used by the acct_gather_profile plugin": "acct_gather_profile 插件使用的剖析配置",
    "Quality of Service assigned to the job": "作业分配的 QoS",
    "Node reboot requested before start": "开始前请求节点重启",
    "Determines whether the job may be requeued": "是否允许作业被重新排队",
    "Name of reservation to use": "要使用的保留资源名称",
    "Job batch script; only the first component in a HetJob is populated or honored": "作业批处理脚本；异构作业中仅首个组件填充/生效",
    "How the job can share resources with other jobs, if at all": "作业与其他作业共享资源的方式（如允许）",
    "Site-specific priority factor": "站点自定义优先级因子",
    "Layout": "布局",
    "Task to TRES binding directives": "任务到 TRES 的绑定指令",
    "TRES frequency directives": "TRES 频率指令",
    "Comma-separated list of TRES=# values to be allocated for every job": "每个作业分配的 TRES=# 列表（逗号分隔）",
    "Comma-separated list of TRES=# values to be allocated for every node": "每个节点分配的 TRES=# 列表（逗号分隔）",
    "Comma-separated list of TRES=# values to be allocated for every socket": "每个插槽分配的 TRES=# 列表（逗号分隔）",
    "Comma-separated list of TRES=# values to be allocated for every task": "每个任务分配的 TRES=# 列表（逗号分隔）",
    "User ID that owns the job": "作业所属用户的 UID",
    "If true, wait to start until after all nodes have booted": "若为 True，则等待所有节点启动后再开始",
    "Flags related to job signals": "与作业信号相关的标志",
    'Signal to send when approaching end time (e.g. "10" or "USR1")': '接近结束时间时发送的信号（如 "10" 或 "USR1"）',
    "Working directory to use for the job": "作业使用的工作目录",
    "Number of CPUs required by each task": "每个任务需要的 CPU 数",
    "Minimum number of CPUs required": "所需 CPU 最小值",
    "Maximum number of CPUs required": "所需 CPU 最大值",
    "Node count range specification (e.g. 1-15:4)": "节点数量范围规范（如 1-15:4）",
    "Minimum node count": "最小节点数",
    "Maximum node count": "最大节点数",
    "Boards per node required": "每节点所需的板卡数",
    "Sockets per board required": "每板所需插槽数",
    "Sockets per node required": "每节点所需插槽数",
    "Threads per core required": "每核心所需线程数",
    "Number of tasks to invoke on each node": "每个节点启动的任务数",
    "Number of tasks to invoke on each socket": "每个插槽启动的任务数",
    "Number of tasks to invoke on each core": "每个核心启动的任务数",
    "Number of tasks to invoke on each board": "每块板卡启动的任务数",
    "Number of tasks that can access each GPU": "可访问每个 GPU 的任务数",
    "Minimum number of CPUs per node": "每节点最少 CPU 数",
    "Minimum tmp disk space required per node": "每节点所需的最小临时磁盘空间",
    "SELinux context": "SELinux 上下文",
    "Path to stderr file": "stderr 文件路径",
    "Path to stdin file": "stdin 文件路径",
    "Path to stdout file": "stdout 文件路径",
    "Maximum time to wait for switches in seconds": "等待交换机的最长时间（秒）",
    "Workload characterization key": "工作负载特征键",
    "X11 forwarding options": "X11 转发选项",
    "Magic cookie for X11 forwarding": "X11 转发的魔术 cookie",
    "Hostname or UNIX socket if x11_target_port=0": "当 x11_target_port=0 时为主机名或 UNIX 套接字",
    "TCP port": "TCP 端口",
}

# ======= Parse fields and comments =======
FIELD_RE = re.compile(
    r"^\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<ann>.+?)\s*=\s*.+?\n"
    r'(?:\s*"""(?P<doc>.+?)"""\s*\n)?',
    re.DOTALL | re.MULTILINE,
)


def clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    oneline = " ".join(doc.split())
    return CN_MAP.get(oneline, oneline)


def infer_click_type(ann: str) -> tuple[str, bool]:
    """
    Return (click_type_str, multiple_flag)
    """
    a = ann.replace(" ", "")
    # list[...] → multiple strings
    if "list[" in a or "List[" in a:
        return "str", True
    # common primitives
    if ":int]" in a or a.endswith("int") or "int]" in a or "int|" in a:
        return "int", False
    if ":bool]" in a or a.endswith("bool") or "bool]" in a or "bool|" in a:
        return "bool", False
    if ":str]" in a or a.endswith("str") or "str]" in a or "str|" in a:
        return "str", False
    # unknown/custom → str
    return "str", False


def to_kebab(name: str) -> str:
    return name.replace("_", "-")


def main():
    matches = list(FIELD_RE.finditer(CLASS_SRC))
    fields = []
    for m in matches:
        name = m.group("name")
        # skip attrs-only non-CLI fields:
        if name in {"additional_properties"}:
            continue
        ann = m.group("ann") or ""
        doc = clean_doc(m.group("doc") or "")
        click_type, multiple = infer_click_type(ann)
        fields.append((name, ann, doc, click_type, multiple))

    # Build Typer CLI source
    imports = ["import json", "import typer", "from typing import Optional, List"]
    header = [
        "app = typer.Typer(help='V0043 Job Submit CLI（根据 V0043JobDescMsg 自动生成）')",
        "",
        "# 说明：",
        "# - 列表参数可通过重复传入，例如：--argv python --argv script.py",
        "# - 未提供的可选项不会出现在最终 JSON 中",
        "",
    ]

    # Build function signature dynamically with **kwargs style (simpler & robust)
    # We'll register options manually inside the command function using Typer Option with defaults.
    # However Typer requires static signature. Instead we create a single command that accepts arbitrary known options
    # by defining them explicitly in the function signature text we generate.
    params_lines = []
    collect_lines = [
        "payload = {}",
    ]

    for name, ann, doc, click_type, multiple in fields:
        opt_name = to_kebab(name)
        help_text = doc or ""
        if click_type == "int":
            default = "None"
            annotation = "Optional[int]"
            option = f"typer.Option(None, help={json_repr(help_text)})"
        elif click_type == "bool":
            # For bool flags, Typer best practice: Option with is_flag
            # But we want tri-state (unset/True/False). We'll accept --flag/--no-flag via bool|None using Option with default None.
            annotation = "Optional[bool]"
            option = f"typer.Option(None, help={json_repr(help_text)})"
        else:  # str
            if multiple:
                annotation = "List[str]"
                option = f"typer.Option(None, help={json_repr(help_text)}, show_default=False)"
                default = "None"
            else:
                annotation = "Optional[str]"
                option = f"typer.Option(None, help={json_repr(help_text)})"
            default = "None"

        # parameter line
        params_lines.append(f"{name}: {annotation} = {option}")
        # collection logic
        if click_type == "bool":
            collect_lines.append(f"if {name} is not None: payload['{name}'] = bool({name})")
        else:
            collect_lines.append(f"if {name} is not None: payload['{name}'] = {name}")

    params_sig = ",\n    ".join(params_lines)
    collect_body = "\n    ".join(collect_lines)

    command_fn = f"""
@app.command('submit')
def submit(
    {params_sig}
):
    \"\"\"提交作业参数 → 输出 JSON（便于上游系统对接）。\"\"\"
    {collect_body}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
"""

    code = (
        "\n".join(imports)
        + "\n\n"
        + "\n".join(header)
        + command_fn
        + "\n\n"
        + "if __name__ == '__main__':\n    app()"
    )

    out = Path("v0043_cli.py")
    out.write_text(code, encoding="utf-8")
    print(f"Generated: {out.resolve()}")


def json_repr(s: str) -> str:
    # produce a Python string literal safely for help texts
    return repr(s)


if __name__ == "__main__":
    main()
