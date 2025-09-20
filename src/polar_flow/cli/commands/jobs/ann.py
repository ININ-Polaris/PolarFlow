submit_ann = {
    "*.job_id": "任务 ID",
    "*.step_id": "已提交步骤 ID",
    "*.job_submit_user_msg": "作业提交用户消息",
}
job_info_ann = {
    # job 基本标识与状态
    "jobs.command": "作业的运行入口",
    "jobs.current_working_directory": "作业的工作目录",
    "jobs.flags": "作业的标志（特性/特征）",
    "jobs.job_id": "作业的 ID，用来唯一标识该作业",
    "jobs.hold": "作业是否暂停",
    "jobs.group_id": "作拥有该作业的用户的组 ID",
    "jobs.group_name": "作拥有该作业的用户的组名称",
    "jobs.job_state": "作业当前的状态",
    "jobs.state_reason": "作业处于当前状态的原因",
    "jobs.tres_req_str": "作业所请求分配的资源",
    "jobs.name": "提交作业时指定的作业名称",
    "jobs.user_name": "提交作业的用户名",
    "jobs.account": "作业所属的账户／项目",
    "jobs.partition": "作业提交或分配所在的分区（partition）",
    "jobs.qos": "作业使用的 QoS（Quality of Service）",
    # 时间／状态
    "jobs.submission_time": "作业被提交的时间戳",
    "jobs.start_time": "作业开始运行的时间；如果未开始则可能为空或缺省",
    "jobs.end_time": "作业结束（完成或失败）的时间；如果未结束则为空",
    "jobs.time_limit": "作业的最大允许运行时间／时限（timelimit）",
    "jobs.state": "当前作业状态（如 PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, etc.）",
    # 资源和规模
    "jobs.nodes": "作业被分配／请求的节点数或节点列表",
    "jobs.ntasks": "任务数（tasks 或进程数）",
    "jobs.cpus_per_task": "每任务所需 CPU 核数",
    "jobs.total_cpus": "作业实际／请求的总 CPU 数（可能等于 ntasks * cpus_per_task）",
    "jobs.memory_per_node": "每个节点分配的内存量（或请求的内存数／限制）",
    "jobs.mem_per_cpu": "每个 CPU 核的内存配额（如果配置了此项）",
    # 输入／输出脚本与环境
    "jobs.script": "提交作业时脚本内容（通常是 sbatch 脚本）",
    "jobs.environment": "作业执行环境变量的列表／映射",
    "jobs.stdout_path": "标准输出文件路径（如果通过选项指定）",
    "jobs.stderr_path": "标准错误文件路径",
    # 依赖与队列行为
    "jobs.dependency": "作业依赖条件（如果有）",
    "jobs.array_job_id": "如果是作业数组 (array job)，这是数组作业的 ID 或标识",
    "jobs.tasks": "更详细的任务子结构，例如任务数，任务在各节点上的分布等",
    # 会计与资源使用等（如果 accounting 开启或 job 已运行完部分）
    "jobs.used_cpus": "已使用的 CPU 总数或已消耗 CPU 时间",
    "jobs.used_memory": "已使用的内存（可能是最大内存／平均内存等）",
    "jobs.used_time": "已运行时间（从开始运行到当前或结束）",
    "jobs.exit_code": "作业完成后的退出状态／代码",
    "jobs.requeue": "是否被重新排队（如果作业失败或被终止后被重试）",
    # 调度与优先级
    "jobs.priority": "作业在调度系统中的优先级值",
    "jobs.nice": "nice 值（如果调度系统使用）",
    "jobs.partition_priority": "所在分区的优先级或排序权重",
    # 节点详情（如果展开）
    "jobs.node_list": "实际分配的节点列表或节点名称范围",
    "jobs.node_features": "节点特性约束（features / constraints）",
    "jobs.constraints": "提交作业时对节点所要求的约束（例如在带特定资源或标签的节点上运行）",
    # 额外／可选字段
    "jobs.comment": "提交或管理员添加的注释字段（如果配置允许）",
    "jobs.submit_host": "从哪个主机提交的作业（login node 或 submission host）",
    "jobs.deadline": "如果作业有 deadline（截止时间）的话",
    # indexing /版本控制
    "jobs.job_array_index": "如果是数组作业，该任务在数组内的索引",
    "jobs.heterogeneous": "是否为异构作业（heterogeneous job）",
    # 可能的信号与取消
    "jobs.cancelled": "作业是否被取消",
    "jobs.cancel_time": "取消时间（如果有）",
    "jobs.fail_reason": "失败原因／错误信息（如果作业状态为失败或取消）",
}
