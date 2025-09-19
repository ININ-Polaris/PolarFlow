from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_job_desc_msg_cpu_binding_flags_item import V0043JobDescMsgCpuBindingFlagsItem
from ..models.v0043_job_desc_msg_flags_item import V0043JobDescMsgFlagsItem
from ..models.v0043_job_desc_msg_kill_warning_flags_item import V0043JobDescMsgKillWarningFlagsItem
from ..models.v0043_job_desc_msg_mail_type_item import V0043JobDescMsgMailTypeItem
from ..models.v0043_job_desc_msg_memory_binding_type_item import V0043JobDescMsgMemoryBindingTypeItem
from ..models.v0043_job_desc_msg_open_mode_item import V0043JobDescMsgOpenModeItem
from ..models.v0043_job_desc_msg_profile_item import V0043JobDescMsgProfileItem
from ..models.v0043_job_desc_msg_shared_item import V0043JobDescMsgSharedItem
from ..models.v0043_job_desc_msg_x11_item import V0043JobDescMsgX11Item
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_cron_entry import V0043CronEntry
    from ..models.v0043_job_desc_msg_rlimits import V0043JobDescMsgRlimits
    from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043JobDescMsg")


@_attrs_define
class V0043JobDescMsg:
    """
    Attributes:
        account (Union[Unset, str]): Account associated with the job
        account_gather_frequency (Union[Unset, str]): Job accounting and profiling sampling intervals in seconds
        admin_comment (Union[Unset, str]): Arbitrary comment made by administrator
        allocation_node_list (Union[Unset, str]): Local node making the resource allocation
        allocation_node_port (Union[Unset, int]): Port to send allocation confirmation to
        argv (Union[Unset, list[str]]):
        array (Union[Unset, str]): Job array index value specification
        batch_features (Union[Unset, str]): Features required for batch script's node
        begin_time (Union[Unset, V0043Uint64NoValStruct]):
        flags (Union[Unset, list[V0043JobDescMsgFlagsItem]]): Job flags
        burst_buffer (Union[Unset, str]): Burst buffer specifications
        clusters (Union[Unset, str]): Clusters that a federated job can run on
        cluster_constraint (Union[Unset, str]): Required features that a federated cluster must have to have a sibling
            job submitted to it
        comment (Union[Unset, str]): Arbitrary comment made by user
        contiguous (Union[Unset, bool]): True if job requires contiguous nodes
        container (Union[Unset, str]): Absolute path to OCI container bundle
        container_id (Union[Unset, str]): OCI container ID
        core_specification (Union[Unset, int]): Specialized core count
        thread_specification (Union[Unset, int]): Specialized thread count
        cpu_binding (Union[Unset, str]): Method for binding tasks to allocated CPUs
        cpu_binding_flags (Union[Unset, list[V0043JobDescMsgCpuBindingFlagsItem]]): Flags for CPU binding
        cpu_frequency (Union[Unset, str]): Requested CPU frequency range <p1>[-p2][:p3]
        cpus_per_tres (Union[Unset, str]): Semicolon delimited list of TRES=# values values indicating how many CPUs
            should be allocated for each specified TRES (currently only used for gres/gpu)
        crontab (Union[Unset, V0043CronEntry]):
        deadline (Union[Unset, int]): Latest time that the job may start (UNIX timestamp) (UNIX timestamp or time string
            recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
        delay_boot (Union[Unset, int]): Number of seconds after job eligible start that nodes will be rebooted to
            satisfy feature specification
        dependency (Union[Unset, str]): Other jobs that must meet certain criteria before this job can start
        end_time (Union[Unset, int]): Expected end time (UNIX timestamp) (UNIX timestamp or time string recognized by
            Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
        environment (Union[Unset, list[str]]):
        rlimits (Union[Unset, V0043JobDescMsgRlimits]):
        excluded_nodes (Union[Unset, list[str]]):
        extra (Union[Unset, str]): Arbitrary string used for node filtering if extra constraints are enabled
        constraints (Union[Unset, str]): Comma-separated list of features that are required
        group_id (Union[Unset, str]): Group ID of the user that owns the job
        hetjob_group (Union[Unset, int]): Unique sequence number applied to this component of the heterogeneous job
        immediate (Union[Unset, bool]): If true, exit if resources are not available within the time period specified
        job_id (Union[Unset, int]): Job ID
        kill_on_node_fail (Union[Unset, bool]): If true, kill job on node failure
        licenses (Union[Unset, str]): License(s) required by the job
        mail_type (Union[Unset, list[V0043JobDescMsgMailTypeItem]]): Mail event type(s)
        mail_user (Union[Unset, str]): User to receive email notifications
        mcs_label (Union[Unset, str]): Multi-Category Security label on the job
        memory_binding (Union[Unset, str]): Binding map for map/mask_cpu
        memory_binding_type (Union[Unset, list[V0043JobDescMsgMemoryBindingTypeItem]]): Method for binding tasks to
            memory
        memory_per_tres (Union[Unset, str]): Semicolon delimited list of TRES=# values indicating how much memory in
            megabytes should be allocated for each specified TRES (currently only used for gres/gpu)
        name (Union[Unset, str]): Job name
        network (Union[Unset, str]): Network specs for job step
        nice (Union[Unset, int]): Requested job priority change
        tasks (Union[Unset, int]): Number of tasks
        oom_kill_step (Union[Unset, int]): Kill whole step in case of OOM in one of the tasks
        open_mode (Union[Unset, list[V0043JobDescMsgOpenModeItem]]): Open mode used for stdout and stderr files
        reserve_ports (Union[Unset, int]): Port to send various notification msg to
        overcommit (Union[Unset, bool]): Overcommit resources
        partition (Union[Unset, str]): Partition assigned to the job
        distribution_plane_size (Union[Unset, V0043Uint16NoValStruct]):
        power_flags (Union[Unset, list[Any]]):
        prefer (Union[Unset, str]): Comma-separated list of features that are preferred but not required
        hold (Union[Unset, bool]): Hold (true) or release (false) job (Job held)
        priority (Union[Unset, V0043Uint32NoValStruct]):
        profile (Union[Unset, list[V0043JobDescMsgProfileItem]]): Profile used by the acct_gather_profile plugin
        qos (Union[Unset, str]): Quality of Service assigned to the job
        reboot (Union[Unset, bool]): Node reboot requested before start
        required_nodes (Union[Unset, list[str]]):
        requeue (Union[Unset, bool]): Determines whether the job may be requeued
        reservation (Union[Unset, str]): Name of reservation to use
        script (Union[Unset, str]): Job batch script; only the first component in a HetJob is populated or honored
        shared (Union[Unset, list[V0043JobDescMsgSharedItem]]): How the job can share resources with other jobs, if at
            all
        site_factor (Union[Unset, int]): Site-specific priority factor
        spank_environment (Union[Unset, list[str]]):
        distribution (Union[Unset, str]): Layout
        time_limit (Union[Unset, V0043Uint32NoValStruct]):
        time_minimum (Union[Unset, V0043Uint32NoValStruct]):
        tres_bind (Union[Unset, str]): Task to TRES binding directives
        tres_freq (Union[Unset, str]): TRES frequency directives
        tres_per_job (Union[Unset, str]): Comma-separated list of TRES=# values to be allocated for every job
        tres_per_node (Union[Unset, str]): Comma-separated list of TRES=# values to be allocated for every node
        tres_per_socket (Union[Unset, str]): Comma-separated list of TRES=# values to be allocated for every socket
        tres_per_task (Union[Unset, str]): Comma-separated list of TRES=# values to be allocated for every task
        user_id (Union[Unset, str]): User ID that owns the job
        wait_all_nodes (Union[Unset, bool]): If true, wait to start until after all nodes have booted
        kill_warning_flags (Union[Unset, list[V0043JobDescMsgKillWarningFlagsItem]]): Flags related to job signals
        kill_warning_signal (Union[Unset, str]): Signal to send when approaching end time (e.g. "10" or "USR1")
        kill_warning_delay (Union[Unset, V0043Uint16NoValStruct]):
        current_working_directory (Union[Unset, str]): Working directory to use for the job
        cpus_per_task (Union[Unset, int]): Number of CPUs required by each task
        minimum_cpus (Union[Unset, int]): Minimum number of CPUs required
        maximum_cpus (Union[Unset, int]): Maximum number of CPUs required
        nodes (Union[Unset, str]): Node count range specification (e.g. 1-15:4)
        minimum_nodes (Union[Unset, int]): Minimum node count
        maximum_nodes (Union[Unset, int]): Maximum node count
        minimum_boards_per_node (Union[Unset, int]): Boards per node required
        minimum_sockets_per_board (Union[Unset, int]): Sockets per board required
        sockets_per_node (Union[Unset, int]): Sockets per node required
        threads_per_core (Union[Unset, int]): Threads per core required
        tasks_per_node (Union[Unset, int]): Number of tasks to invoke on each node
        tasks_per_socket (Union[Unset, int]): Number of tasks to invoke on each socket
        tasks_per_core (Union[Unset, int]): Number of tasks to invoke on each core
        tasks_per_board (Union[Unset, int]): Number of tasks to invoke on each board
        ntasks_per_tres (Union[Unset, int]): Number of tasks that can access each GPU
        minimum_cpus_per_node (Union[Unset, int]): Minimum number of CPUs per node
        memory_per_cpu (Union[Unset, V0043Uint64NoValStruct]):
        memory_per_node (Union[Unset, V0043Uint64NoValStruct]):
        temporary_disk_per_node (Union[Unset, int]): Minimum tmp disk space required per node
        selinux_context (Union[Unset, str]): SELinux context
        required_switches (Union[Unset, V0043Uint32NoValStruct]):
        segment_size (Union[Unset, V0043Uint16NoValStruct]):
        standard_error (Union[Unset, str]): Path to stderr file
        standard_input (Union[Unset, str]): Path to stdin file
        standard_output (Union[Unset, str]): Path to stdout file
        wait_for_switch (Union[Unset, int]): Maximum time to wait for switches in seconds
        wckey (Union[Unset, str]): Workload characterization key
        x11 (Union[Unset, list[V0043JobDescMsgX11Item]]): X11 forwarding options
        x11_magic_cookie (Union[Unset, str]): Magic cookie for X11 forwarding
        x11_target_host (Union[Unset, str]): Hostname or UNIX socket if x11_target_port=0
        x11_target_port (Union[Unset, int]): TCP port
    """

    account: Union[Unset, str] = UNSET
    account_gather_frequency: Union[Unset, str] = UNSET
    admin_comment: Union[Unset, str] = UNSET
    allocation_node_list: Union[Unset, str] = UNSET
    allocation_node_port: Union[Unset, int] = UNSET
    argv: Union[Unset, list[str]] = UNSET
    array: Union[Unset, str] = UNSET
    batch_features: Union[Unset, str] = UNSET
    begin_time: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    flags: Union[Unset, list[V0043JobDescMsgFlagsItem]] = UNSET
    burst_buffer: Union[Unset, str] = UNSET
    clusters: Union[Unset, str] = UNSET
    cluster_constraint: Union[Unset, str] = UNSET
    comment: Union[Unset, str] = UNSET
    contiguous: Union[Unset, bool] = UNSET
    container: Union[Unset, str] = UNSET
    container_id: Union[Unset, str] = UNSET
    core_specification: Union[Unset, int] = UNSET
    thread_specification: Union[Unset, int] = UNSET
    cpu_binding: Union[Unset, str] = UNSET
    cpu_binding_flags: Union[Unset, list[V0043JobDescMsgCpuBindingFlagsItem]] = UNSET
    cpu_frequency: Union[Unset, str] = UNSET
    cpus_per_tres: Union[Unset, str] = UNSET
    crontab: Union[Unset, "V0043CronEntry"] = UNSET
    deadline: Union[Unset, int] = UNSET
    delay_boot: Union[Unset, int] = UNSET
    dependency: Union[Unset, str] = UNSET
    end_time: Union[Unset, int] = UNSET
    environment: Union[Unset, list[str]] = UNSET
    rlimits: Union[Unset, "V0043JobDescMsgRlimits"] = UNSET
    excluded_nodes: Union[Unset, list[str]] = UNSET
    extra: Union[Unset, str] = UNSET
    constraints: Union[Unset, str] = UNSET
    group_id: Union[Unset, str] = UNSET
    hetjob_group: Union[Unset, int] = UNSET
    immediate: Union[Unset, bool] = UNSET
    job_id: Union[Unset, int] = UNSET
    kill_on_node_fail: Union[Unset, bool] = UNSET
    licenses: Union[Unset, str] = UNSET
    mail_type: Union[Unset, list[V0043JobDescMsgMailTypeItem]] = UNSET
    mail_user: Union[Unset, str] = UNSET
    mcs_label: Union[Unset, str] = UNSET
    memory_binding: Union[Unset, str] = UNSET
    memory_binding_type: Union[Unset, list[V0043JobDescMsgMemoryBindingTypeItem]] = UNSET
    memory_per_tres: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    network: Union[Unset, str] = UNSET
    nice: Union[Unset, int] = UNSET
    tasks: Union[Unset, int] = UNSET
    oom_kill_step: Union[Unset, int] = UNSET
    open_mode: Union[Unset, list[V0043JobDescMsgOpenModeItem]] = UNSET
    reserve_ports: Union[Unset, int] = UNSET
    overcommit: Union[Unset, bool] = UNSET
    partition: Union[Unset, str] = UNSET
    distribution_plane_size: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    power_flags: Union[Unset, list[Any]] = UNSET
    prefer: Union[Unset, str] = UNSET
    hold: Union[Unset, bool] = UNSET
    priority: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    profile: Union[Unset, list[V0043JobDescMsgProfileItem]] = UNSET
    qos: Union[Unset, str] = UNSET
    reboot: Union[Unset, bool] = UNSET
    required_nodes: Union[Unset, list[str]] = UNSET
    requeue: Union[Unset, bool] = UNSET
    reservation: Union[Unset, str] = UNSET
    script: Union[Unset, str] = UNSET
    shared: Union[Unset, list[V0043JobDescMsgSharedItem]] = UNSET
    site_factor: Union[Unset, int] = UNSET
    spank_environment: Union[Unset, list[str]] = UNSET
    distribution: Union[Unset, str] = UNSET
    time_limit: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    time_minimum: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    tres_bind: Union[Unset, str] = UNSET
    tres_freq: Union[Unset, str] = UNSET
    tres_per_job: Union[Unset, str] = UNSET
    tres_per_node: Union[Unset, str] = UNSET
    tres_per_socket: Union[Unset, str] = UNSET
    tres_per_task: Union[Unset, str] = UNSET
    user_id: Union[Unset, str] = UNSET
    wait_all_nodes: Union[Unset, bool] = UNSET
    kill_warning_flags: Union[Unset, list[V0043JobDescMsgKillWarningFlagsItem]] = UNSET
    kill_warning_signal: Union[Unset, str] = UNSET
    kill_warning_delay: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    current_working_directory: Union[Unset, str] = UNSET
    cpus_per_task: Union[Unset, int] = UNSET
    minimum_cpus: Union[Unset, int] = UNSET
    maximum_cpus: Union[Unset, int] = UNSET
    nodes: Union[Unset, str] = UNSET
    minimum_nodes: Union[Unset, int] = UNSET
    maximum_nodes: Union[Unset, int] = UNSET
    minimum_boards_per_node: Union[Unset, int] = UNSET
    minimum_sockets_per_board: Union[Unset, int] = UNSET
    sockets_per_node: Union[Unset, int] = UNSET
    threads_per_core: Union[Unset, int] = UNSET
    tasks_per_node: Union[Unset, int] = UNSET
    tasks_per_socket: Union[Unset, int] = UNSET
    tasks_per_core: Union[Unset, int] = UNSET
    tasks_per_board: Union[Unset, int] = UNSET
    ntasks_per_tres: Union[Unset, int] = UNSET
    minimum_cpus_per_node: Union[Unset, int] = UNSET
    memory_per_cpu: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    memory_per_node: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    temporary_disk_per_node: Union[Unset, int] = UNSET
    selinux_context: Union[Unset, str] = UNSET
    required_switches: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    segment_size: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    standard_error: Union[Unset, str] = UNSET
    standard_input: Union[Unset, str] = UNSET
    standard_output: Union[Unset, str] = UNSET
    wait_for_switch: Union[Unset, int] = UNSET
    wckey: Union[Unset, str] = UNSET
    x11: Union[Unset, list[V0043JobDescMsgX11Item]] = UNSET
    x11_magic_cookie: Union[Unset, str] = UNSET
    x11_target_host: Union[Unset, str] = UNSET
    x11_target_port: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        account = self.account

        account_gather_frequency = self.account_gather_frequency

        admin_comment = self.admin_comment

        allocation_node_list = self.allocation_node_list

        allocation_node_port = self.allocation_node_port

        argv: Union[Unset, list[str]] = UNSET
        if not isinstance(self.argv, Unset):
            argv = self.argv

        array = self.array

        batch_features = self.batch_features

        begin_time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.begin_time, Unset):
            begin_time = self.begin_time.to_dict()

        flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.flags, Unset):
            flags = []
            for flags_item_data in self.flags:
                flags_item = flags_item_data.value
                flags.append(flags_item)

        burst_buffer = self.burst_buffer

        clusters = self.clusters

        cluster_constraint = self.cluster_constraint

        comment = self.comment

        contiguous = self.contiguous

        container = self.container

        container_id = self.container_id

        core_specification = self.core_specification

        thread_specification = self.thread_specification

        cpu_binding = self.cpu_binding

        cpu_binding_flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.cpu_binding_flags, Unset):
            cpu_binding_flags = []
            for cpu_binding_flags_item_data in self.cpu_binding_flags:
                cpu_binding_flags_item = cpu_binding_flags_item_data.value
                cpu_binding_flags.append(cpu_binding_flags_item)

        cpu_frequency = self.cpu_frequency

        cpus_per_tres = self.cpus_per_tres

        crontab: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.crontab, Unset):
            crontab = self.crontab.to_dict()

        deadline = self.deadline

        delay_boot = self.delay_boot

        dependency = self.dependency

        end_time = self.end_time

        environment: Union[Unset, list[str]] = UNSET
        if not isinstance(self.environment, Unset):
            environment = self.environment

        rlimits: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.rlimits, Unset):
            rlimits = self.rlimits.to_dict()

        excluded_nodes: Union[Unset, list[str]] = UNSET
        if not isinstance(self.excluded_nodes, Unset):
            excluded_nodes = self.excluded_nodes

        extra = self.extra

        constraints = self.constraints

        group_id = self.group_id

        hetjob_group = self.hetjob_group

        immediate = self.immediate

        job_id = self.job_id

        kill_on_node_fail = self.kill_on_node_fail

        licenses = self.licenses

        mail_type: Union[Unset, list[str]] = UNSET
        if not isinstance(self.mail_type, Unset):
            mail_type = []
            for mail_type_item_data in self.mail_type:
                mail_type_item = mail_type_item_data.value
                mail_type.append(mail_type_item)

        mail_user = self.mail_user

        mcs_label = self.mcs_label

        memory_binding = self.memory_binding

        memory_binding_type: Union[Unset, list[str]] = UNSET
        if not isinstance(self.memory_binding_type, Unset):
            memory_binding_type = []
            for memory_binding_type_item_data in self.memory_binding_type:
                memory_binding_type_item = memory_binding_type_item_data.value
                memory_binding_type.append(memory_binding_type_item)

        memory_per_tres = self.memory_per_tres

        name = self.name

        network = self.network

        nice = self.nice

        tasks = self.tasks

        oom_kill_step = self.oom_kill_step

        open_mode: Union[Unset, list[str]] = UNSET
        if not isinstance(self.open_mode, Unset):
            open_mode = []
            for open_mode_item_data in self.open_mode:
                open_mode_item = open_mode_item_data.value
                open_mode.append(open_mode_item)

        reserve_ports = self.reserve_ports

        overcommit = self.overcommit

        partition = self.partition

        distribution_plane_size: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.distribution_plane_size, Unset):
            distribution_plane_size = self.distribution_plane_size.to_dict()

        power_flags: Union[Unset, list[Any]] = UNSET
        if not isinstance(self.power_flags, Unset):
            power_flags = self.power_flags

        prefer = self.prefer

        hold = self.hold

        priority: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.priority, Unset):
            priority = self.priority.to_dict()

        profile: Union[Unset, list[str]] = UNSET
        if not isinstance(self.profile, Unset):
            profile = []
            for profile_item_data in self.profile:
                profile_item = profile_item_data.value
                profile.append(profile_item)

        qos = self.qos

        reboot = self.reboot

        required_nodes: Union[Unset, list[str]] = UNSET
        if not isinstance(self.required_nodes, Unset):
            required_nodes = self.required_nodes

        requeue = self.requeue

        reservation = self.reservation

        script = self.script

        shared: Union[Unset, list[str]] = UNSET
        if not isinstance(self.shared, Unset):
            shared = []
            for shared_item_data in self.shared:
                shared_item = shared_item_data.value
                shared.append(shared_item)

        site_factor = self.site_factor

        spank_environment: Union[Unset, list[str]] = UNSET
        if not isinstance(self.spank_environment, Unset):
            spank_environment = self.spank_environment

        distribution = self.distribution

        time_limit: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.time_limit, Unset):
            time_limit = self.time_limit.to_dict()

        time_minimum: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.time_minimum, Unset):
            time_minimum = self.time_minimum.to_dict()

        tres_bind = self.tres_bind

        tres_freq = self.tres_freq

        tres_per_job = self.tres_per_job

        tres_per_node = self.tres_per_node

        tres_per_socket = self.tres_per_socket

        tres_per_task = self.tres_per_task

        user_id = self.user_id

        wait_all_nodes = self.wait_all_nodes

        kill_warning_flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.kill_warning_flags, Unset):
            kill_warning_flags = []
            for kill_warning_flags_item_data in self.kill_warning_flags:
                kill_warning_flags_item = kill_warning_flags_item_data.value
                kill_warning_flags.append(kill_warning_flags_item)

        kill_warning_signal = self.kill_warning_signal

        kill_warning_delay: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.kill_warning_delay, Unset):
            kill_warning_delay = self.kill_warning_delay.to_dict()

        current_working_directory = self.current_working_directory

        cpus_per_task = self.cpus_per_task

        minimum_cpus = self.minimum_cpus

        maximum_cpus = self.maximum_cpus

        nodes = self.nodes

        minimum_nodes = self.minimum_nodes

        maximum_nodes = self.maximum_nodes

        minimum_boards_per_node = self.minimum_boards_per_node

        minimum_sockets_per_board = self.minimum_sockets_per_board

        sockets_per_node = self.sockets_per_node

        threads_per_core = self.threads_per_core

        tasks_per_node = self.tasks_per_node

        tasks_per_socket = self.tasks_per_socket

        tasks_per_core = self.tasks_per_core

        tasks_per_board = self.tasks_per_board

        ntasks_per_tres = self.ntasks_per_tres

        minimum_cpus_per_node = self.minimum_cpus_per_node

        memory_per_cpu: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.memory_per_cpu, Unset):
            memory_per_cpu = self.memory_per_cpu.to_dict()

        memory_per_node: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.memory_per_node, Unset):
            memory_per_node = self.memory_per_node.to_dict()

        temporary_disk_per_node = self.temporary_disk_per_node

        selinux_context = self.selinux_context

        required_switches: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.required_switches, Unset):
            required_switches = self.required_switches.to_dict()

        segment_size: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.segment_size, Unset):
            segment_size = self.segment_size.to_dict()

        standard_error = self.standard_error

        standard_input = self.standard_input

        standard_output = self.standard_output

        wait_for_switch = self.wait_for_switch

        wckey = self.wckey

        x11: Union[Unset, list[str]] = UNSET
        if not isinstance(self.x11, Unset):
            x11 = []
            for x11_item_data in self.x11:
                x11_item = x11_item_data.value
                x11.append(x11_item)

        x11_magic_cookie = self.x11_magic_cookie

        x11_target_host = self.x11_target_host

        x11_target_port = self.x11_target_port

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if account is not UNSET:
            field_dict["account"] = account
        if account_gather_frequency is not UNSET:
            field_dict["account_gather_frequency"] = account_gather_frequency
        if admin_comment is not UNSET:
            field_dict["admin_comment"] = admin_comment
        if allocation_node_list is not UNSET:
            field_dict["allocation_node_list"] = allocation_node_list
        if allocation_node_port is not UNSET:
            field_dict["allocation_node_port"] = allocation_node_port
        if argv is not UNSET:
            field_dict["argv"] = argv
        if array is not UNSET:
            field_dict["array"] = array
        if batch_features is not UNSET:
            field_dict["batch_features"] = batch_features
        if begin_time is not UNSET:
            field_dict["begin_time"] = begin_time
        if flags is not UNSET:
            field_dict["flags"] = flags
        if burst_buffer is not UNSET:
            field_dict["burst_buffer"] = burst_buffer
        if clusters is not UNSET:
            field_dict["clusters"] = clusters
        if cluster_constraint is not UNSET:
            field_dict["cluster_constraint"] = cluster_constraint
        if comment is not UNSET:
            field_dict["comment"] = comment
        if contiguous is not UNSET:
            field_dict["contiguous"] = contiguous
        if container is not UNSET:
            field_dict["container"] = container
        if container_id is not UNSET:
            field_dict["container_id"] = container_id
        if core_specification is not UNSET:
            field_dict["core_specification"] = core_specification
        if thread_specification is not UNSET:
            field_dict["thread_specification"] = thread_specification
        if cpu_binding is not UNSET:
            field_dict["cpu_binding"] = cpu_binding
        if cpu_binding_flags is not UNSET:
            field_dict["cpu_binding_flags"] = cpu_binding_flags
        if cpu_frequency is not UNSET:
            field_dict["cpu_frequency"] = cpu_frequency
        if cpus_per_tres is not UNSET:
            field_dict["cpus_per_tres"] = cpus_per_tres
        if crontab is not UNSET:
            field_dict["crontab"] = crontab
        if deadline is not UNSET:
            field_dict["deadline"] = deadline
        if delay_boot is not UNSET:
            field_dict["delay_boot"] = delay_boot
        if dependency is not UNSET:
            field_dict["dependency"] = dependency
        if end_time is not UNSET:
            field_dict["end_time"] = end_time
        if environment is not UNSET:
            field_dict["environment"] = environment
        if rlimits is not UNSET:
            field_dict["rlimits"] = rlimits
        if excluded_nodes is not UNSET:
            field_dict["excluded_nodes"] = excluded_nodes
        if extra is not UNSET:
            field_dict["extra"] = extra
        if constraints is not UNSET:
            field_dict["constraints"] = constraints
        if group_id is not UNSET:
            field_dict["group_id"] = group_id
        if hetjob_group is not UNSET:
            field_dict["hetjob_group"] = hetjob_group
        if immediate is not UNSET:
            field_dict["immediate"] = immediate
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if kill_on_node_fail is not UNSET:
            field_dict["kill_on_node_fail"] = kill_on_node_fail
        if licenses is not UNSET:
            field_dict["licenses"] = licenses
        if mail_type is not UNSET:
            field_dict["mail_type"] = mail_type
        if mail_user is not UNSET:
            field_dict["mail_user"] = mail_user
        if mcs_label is not UNSET:
            field_dict["mcs_label"] = mcs_label
        if memory_binding is not UNSET:
            field_dict["memory_binding"] = memory_binding
        if memory_binding_type is not UNSET:
            field_dict["memory_binding_type"] = memory_binding_type
        if memory_per_tres is not UNSET:
            field_dict["memory_per_tres"] = memory_per_tres
        if name is not UNSET:
            field_dict["name"] = name
        if network is not UNSET:
            field_dict["network"] = network
        if nice is not UNSET:
            field_dict["nice"] = nice
        if tasks is not UNSET:
            field_dict["tasks"] = tasks
        if oom_kill_step is not UNSET:
            field_dict["oom_kill_step"] = oom_kill_step
        if open_mode is not UNSET:
            field_dict["open_mode"] = open_mode
        if reserve_ports is not UNSET:
            field_dict["reserve_ports"] = reserve_ports
        if overcommit is not UNSET:
            field_dict["overcommit"] = overcommit
        if partition is not UNSET:
            field_dict["partition"] = partition
        if distribution_plane_size is not UNSET:
            field_dict["distribution_plane_size"] = distribution_plane_size
        if power_flags is not UNSET:
            field_dict["power_flags"] = power_flags
        if prefer is not UNSET:
            field_dict["prefer"] = prefer
        if hold is not UNSET:
            field_dict["hold"] = hold
        if priority is not UNSET:
            field_dict["priority"] = priority
        if profile is not UNSET:
            field_dict["profile"] = profile
        if qos is not UNSET:
            field_dict["qos"] = qos
        if reboot is not UNSET:
            field_dict["reboot"] = reboot
        if required_nodes is not UNSET:
            field_dict["required_nodes"] = required_nodes
        if requeue is not UNSET:
            field_dict["requeue"] = requeue
        if reservation is not UNSET:
            field_dict["reservation"] = reservation
        if script is not UNSET:
            field_dict["script"] = script
        if shared is not UNSET:
            field_dict["shared"] = shared
        if site_factor is not UNSET:
            field_dict["site_factor"] = site_factor
        if spank_environment is not UNSET:
            field_dict["spank_environment"] = spank_environment
        if distribution is not UNSET:
            field_dict["distribution"] = distribution
        if time_limit is not UNSET:
            field_dict["time_limit"] = time_limit
        if time_minimum is not UNSET:
            field_dict["time_minimum"] = time_minimum
        if tres_bind is not UNSET:
            field_dict["tres_bind"] = tres_bind
        if tres_freq is not UNSET:
            field_dict["tres_freq"] = tres_freq
        if tres_per_job is not UNSET:
            field_dict["tres_per_job"] = tres_per_job
        if tres_per_node is not UNSET:
            field_dict["tres_per_node"] = tres_per_node
        if tres_per_socket is not UNSET:
            field_dict["tres_per_socket"] = tres_per_socket
        if tres_per_task is not UNSET:
            field_dict["tres_per_task"] = tres_per_task
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if wait_all_nodes is not UNSET:
            field_dict["wait_all_nodes"] = wait_all_nodes
        if kill_warning_flags is not UNSET:
            field_dict["kill_warning_flags"] = kill_warning_flags
        if kill_warning_signal is not UNSET:
            field_dict["kill_warning_signal"] = kill_warning_signal
        if kill_warning_delay is not UNSET:
            field_dict["kill_warning_delay"] = kill_warning_delay
        if current_working_directory is not UNSET:
            field_dict["current_working_directory"] = current_working_directory
        if cpus_per_task is not UNSET:
            field_dict["cpus_per_task"] = cpus_per_task
        if minimum_cpus is not UNSET:
            field_dict["minimum_cpus"] = minimum_cpus
        if maximum_cpus is not UNSET:
            field_dict["maximum_cpus"] = maximum_cpus
        if nodes is not UNSET:
            field_dict["nodes"] = nodes
        if minimum_nodes is not UNSET:
            field_dict["minimum_nodes"] = minimum_nodes
        if maximum_nodes is not UNSET:
            field_dict["maximum_nodes"] = maximum_nodes
        if minimum_boards_per_node is not UNSET:
            field_dict["minimum_boards_per_node"] = minimum_boards_per_node
        if minimum_sockets_per_board is not UNSET:
            field_dict["minimum_sockets_per_board"] = minimum_sockets_per_board
        if sockets_per_node is not UNSET:
            field_dict["sockets_per_node"] = sockets_per_node
        if threads_per_core is not UNSET:
            field_dict["threads_per_core"] = threads_per_core
        if tasks_per_node is not UNSET:
            field_dict["tasks_per_node"] = tasks_per_node
        if tasks_per_socket is not UNSET:
            field_dict["tasks_per_socket"] = tasks_per_socket
        if tasks_per_core is not UNSET:
            field_dict["tasks_per_core"] = tasks_per_core
        if tasks_per_board is not UNSET:
            field_dict["tasks_per_board"] = tasks_per_board
        if ntasks_per_tres is not UNSET:
            field_dict["ntasks_per_tres"] = ntasks_per_tres
        if minimum_cpus_per_node is not UNSET:
            field_dict["minimum_cpus_per_node"] = minimum_cpus_per_node
        if memory_per_cpu is not UNSET:
            field_dict["memory_per_cpu"] = memory_per_cpu
        if memory_per_node is not UNSET:
            field_dict["memory_per_node"] = memory_per_node
        if temporary_disk_per_node is not UNSET:
            field_dict["temporary_disk_per_node"] = temporary_disk_per_node
        if selinux_context is not UNSET:
            field_dict["selinux_context"] = selinux_context
        if required_switches is not UNSET:
            field_dict["required_switches"] = required_switches
        if segment_size is not UNSET:
            field_dict["segment_size"] = segment_size
        if standard_error is not UNSET:
            field_dict["standard_error"] = standard_error
        if standard_input is not UNSET:
            field_dict["standard_input"] = standard_input
        if standard_output is not UNSET:
            field_dict["standard_output"] = standard_output
        if wait_for_switch is not UNSET:
            field_dict["wait_for_switch"] = wait_for_switch
        if wckey is not UNSET:
            field_dict["wckey"] = wckey
        if x11 is not UNSET:
            field_dict["x11"] = x11
        if x11_magic_cookie is not UNSET:
            field_dict["x11_magic_cookie"] = x11_magic_cookie
        if x11_target_host is not UNSET:
            field_dict["x11_target_host"] = x11_target_host
        if x11_target_port is not UNSET:
            field_dict["x11_target_port"] = x11_target_port

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_cron_entry import V0043CronEntry
        from ..models.v0043_job_desc_msg_rlimits import V0043JobDescMsgRlimits
        from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        account = d.pop("account", UNSET)

        account_gather_frequency = d.pop("account_gather_frequency", UNSET)

        admin_comment = d.pop("admin_comment", UNSET)

        allocation_node_list = d.pop("allocation_node_list", UNSET)

        allocation_node_port = d.pop("allocation_node_port", UNSET)

        argv = cast(list[str], d.pop("argv", UNSET))

        array = d.pop("array", UNSET)

        batch_features = d.pop("batch_features", UNSET)

        _begin_time = d.pop("begin_time", UNSET)
        begin_time: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_begin_time, Unset):
            begin_time = UNSET
        else:
            begin_time = V0043Uint64NoValStruct.from_dict(_begin_time)

        flags = []
        _flags = d.pop("flags", UNSET)
        for flags_item_data in _flags or []:
            flags_item = V0043JobDescMsgFlagsItem(flags_item_data)

            flags.append(flags_item)

        burst_buffer = d.pop("burst_buffer", UNSET)

        clusters = d.pop("clusters", UNSET)

        cluster_constraint = d.pop("cluster_constraint", UNSET)

        comment = d.pop("comment", UNSET)

        contiguous = d.pop("contiguous", UNSET)

        container = d.pop("container", UNSET)

        container_id = d.pop("container_id", UNSET)

        core_specification = d.pop("core_specification", UNSET)

        thread_specification = d.pop("thread_specification", UNSET)

        cpu_binding = d.pop("cpu_binding", UNSET)

        cpu_binding_flags = []
        _cpu_binding_flags = d.pop("cpu_binding_flags", UNSET)
        for cpu_binding_flags_item_data in _cpu_binding_flags or []:
            cpu_binding_flags_item = V0043JobDescMsgCpuBindingFlagsItem(cpu_binding_flags_item_data)

            cpu_binding_flags.append(cpu_binding_flags_item)

        cpu_frequency = d.pop("cpu_frequency", UNSET)

        cpus_per_tres = d.pop("cpus_per_tres", UNSET)

        _crontab = d.pop("crontab", UNSET)
        crontab: Union[Unset, V0043CronEntry]
        if isinstance(_crontab, Unset):
            crontab = UNSET
        else:
            crontab = V0043CronEntry.from_dict(_crontab)

        deadline = d.pop("deadline", UNSET)

        delay_boot = d.pop("delay_boot", UNSET)

        dependency = d.pop("dependency", UNSET)

        end_time = d.pop("end_time", UNSET)

        environment = cast(list[str], d.pop("environment", UNSET))

        _rlimits = d.pop("rlimits", UNSET)
        rlimits: Union[Unset, V0043JobDescMsgRlimits]
        if isinstance(_rlimits, Unset):
            rlimits = UNSET
        else:
            rlimits = V0043JobDescMsgRlimits.from_dict(_rlimits)

        excluded_nodes = cast(list[str], d.pop("excluded_nodes", UNSET))

        extra = d.pop("extra", UNSET)

        constraints = d.pop("constraints", UNSET)

        group_id = d.pop("group_id", UNSET)

        hetjob_group = d.pop("hetjob_group", UNSET)

        immediate = d.pop("immediate", UNSET)

        job_id = d.pop("job_id", UNSET)

        kill_on_node_fail = d.pop("kill_on_node_fail", UNSET)

        licenses = d.pop("licenses", UNSET)

        mail_type = []
        _mail_type = d.pop("mail_type", UNSET)
        for mail_type_item_data in _mail_type or []:
            mail_type_item = V0043JobDescMsgMailTypeItem(mail_type_item_data)

            mail_type.append(mail_type_item)

        mail_user = d.pop("mail_user", UNSET)

        mcs_label = d.pop("mcs_label", UNSET)

        memory_binding = d.pop("memory_binding", UNSET)

        memory_binding_type = []
        _memory_binding_type = d.pop("memory_binding_type", UNSET)
        for memory_binding_type_item_data in _memory_binding_type or []:
            memory_binding_type_item = V0043JobDescMsgMemoryBindingTypeItem(memory_binding_type_item_data)

            memory_binding_type.append(memory_binding_type_item)

        memory_per_tres = d.pop("memory_per_tres", UNSET)

        name = d.pop("name", UNSET)

        network = d.pop("network", UNSET)

        nice = d.pop("nice", UNSET)

        tasks = d.pop("tasks", UNSET)

        oom_kill_step = d.pop("oom_kill_step", UNSET)

        open_mode = []
        _open_mode = d.pop("open_mode", UNSET)
        for open_mode_item_data in _open_mode or []:
            open_mode_item = V0043JobDescMsgOpenModeItem(open_mode_item_data)

            open_mode.append(open_mode_item)

        reserve_ports = d.pop("reserve_ports", UNSET)

        overcommit = d.pop("overcommit", UNSET)

        partition = d.pop("partition", UNSET)

        _distribution_plane_size = d.pop("distribution_plane_size", UNSET)
        distribution_plane_size: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_distribution_plane_size, Unset):
            distribution_plane_size = UNSET
        else:
            distribution_plane_size = V0043Uint16NoValStruct.from_dict(_distribution_plane_size)

        power_flags = cast(list[Any], d.pop("power_flags", UNSET))

        prefer = d.pop("prefer", UNSET)

        hold = d.pop("hold", UNSET)

        _priority = d.pop("priority", UNSET)
        priority: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_priority, Unset):
            priority = UNSET
        else:
            priority = V0043Uint32NoValStruct.from_dict(_priority)

        profile = []
        _profile = d.pop("profile", UNSET)
        for profile_item_data in _profile or []:
            profile_item = V0043JobDescMsgProfileItem(profile_item_data)

            profile.append(profile_item)

        qos = d.pop("qos", UNSET)

        reboot = d.pop("reboot", UNSET)

        required_nodes = cast(list[str], d.pop("required_nodes", UNSET))

        requeue = d.pop("requeue", UNSET)

        reservation = d.pop("reservation", UNSET)

        script = d.pop("script", UNSET)

        shared = []
        _shared = d.pop("shared", UNSET)
        for shared_item_data in _shared or []:
            shared_item = V0043JobDescMsgSharedItem(shared_item_data)

            shared.append(shared_item)

        site_factor = d.pop("site_factor", UNSET)

        spank_environment = cast(list[str], d.pop("spank_environment", UNSET))

        distribution = d.pop("distribution", UNSET)

        _time_limit = d.pop("time_limit", UNSET)
        time_limit: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_time_limit, Unset):
            time_limit = UNSET
        else:
            time_limit = V0043Uint32NoValStruct.from_dict(_time_limit)

        _time_minimum = d.pop("time_minimum", UNSET)
        time_minimum: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_time_minimum, Unset):
            time_minimum = UNSET
        else:
            time_minimum = V0043Uint32NoValStruct.from_dict(_time_minimum)

        tres_bind = d.pop("tres_bind", UNSET)

        tres_freq = d.pop("tres_freq", UNSET)

        tres_per_job = d.pop("tres_per_job", UNSET)

        tres_per_node = d.pop("tres_per_node", UNSET)

        tres_per_socket = d.pop("tres_per_socket", UNSET)

        tres_per_task = d.pop("tres_per_task", UNSET)

        user_id = d.pop("user_id", UNSET)

        wait_all_nodes = d.pop("wait_all_nodes", UNSET)

        kill_warning_flags = []
        _kill_warning_flags = d.pop("kill_warning_flags", UNSET)
        for kill_warning_flags_item_data in _kill_warning_flags or []:
            kill_warning_flags_item = V0043JobDescMsgKillWarningFlagsItem(kill_warning_flags_item_data)

            kill_warning_flags.append(kill_warning_flags_item)

        kill_warning_signal = d.pop("kill_warning_signal", UNSET)

        _kill_warning_delay = d.pop("kill_warning_delay", UNSET)
        kill_warning_delay: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_kill_warning_delay, Unset):
            kill_warning_delay = UNSET
        else:
            kill_warning_delay = V0043Uint16NoValStruct.from_dict(_kill_warning_delay)

        current_working_directory = d.pop("current_working_directory", UNSET)

        cpus_per_task = d.pop("cpus_per_task", UNSET)

        minimum_cpus = d.pop("minimum_cpus", UNSET)

        maximum_cpus = d.pop("maximum_cpus", UNSET)

        nodes = d.pop("nodes", UNSET)

        minimum_nodes = d.pop("minimum_nodes", UNSET)

        maximum_nodes = d.pop("maximum_nodes", UNSET)

        minimum_boards_per_node = d.pop("minimum_boards_per_node", UNSET)

        minimum_sockets_per_board = d.pop("minimum_sockets_per_board", UNSET)

        sockets_per_node = d.pop("sockets_per_node", UNSET)

        threads_per_core = d.pop("threads_per_core", UNSET)

        tasks_per_node = d.pop("tasks_per_node", UNSET)

        tasks_per_socket = d.pop("tasks_per_socket", UNSET)

        tasks_per_core = d.pop("tasks_per_core", UNSET)

        tasks_per_board = d.pop("tasks_per_board", UNSET)

        ntasks_per_tres = d.pop("ntasks_per_tres", UNSET)

        minimum_cpus_per_node = d.pop("minimum_cpus_per_node", UNSET)

        _memory_per_cpu = d.pop("memory_per_cpu", UNSET)
        memory_per_cpu: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_memory_per_cpu, Unset):
            memory_per_cpu = UNSET
        else:
            memory_per_cpu = V0043Uint64NoValStruct.from_dict(_memory_per_cpu)

        _memory_per_node = d.pop("memory_per_node", UNSET)
        memory_per_node: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_memory_per_node, Unset):
            memory_per_node = UNSET
        else:
            memory_per_node = V0043Uint64NoValStruct.from_dict(_memory_per_node)

        temporary_disk_per_node = d.pop("temporary_disk_per_node", UNSET)

        selinux_context = d.pop("selinux_context", UNSET)

        _required_switches = d.pop("required_switches", UNSET)
        required_switches: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_required_switches, Unset):
            required_switches = UNSET
        else:
            required_switches = V0043Uint32NoValStruct.from_dict(_required_switches)

        _segment_size = d.pop("segment_size", UNSET)
        segment_size: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_segment_size, Unset):
            segment_size = UNSET
        else:
            segment_size = V0043Uint16NoValStruct.from_dict(_segment_size)

        standard_error = d.pop("standard_error", UNSET)

        standard_input = d.pop("standard_input", UNSET)

        standard_output = d.pop("standard_output", UNSET)

        wait_for_switch = d.pop("wait_for_switch", UNSET)

        wckey = d.pop("wckey", UNSET)

        x11 = []
        _x11 = d.pop("x11", UNSET)
        for x11_item_data in _x11 or []:
            x11_item = V0043JobDescMsgX11Item(x11_item_data)

            x11.append(x11_item)

        x11_magic_cookie = d.pop("x11_magic_cookie", UNSET)

        x11_target_host = d.pop("x11_target_host", UNSET)

        x11_target_port = d.pop("x11_target_port", UNSET)

        v0043_job_desc_msg = cls(
            account=account,
            account_gather_frequency=account_gather_frequency,
            admin_comment=admin_comment,
            allocation_node_list=allocation_node_list,
            allocation_node_port=allocation_node_port,
            argv=argv,
            array=array,
            batch_features=batch_features,
            begin_time=begin_time,
            flags=flags,
            burst_buffer=burst_buffer,
            clusters=clusters,
            cluster_constraint=cluster_constraint,
            comment=comment,
            contiguous=contiguous,
            container=container,
            container_id=container_id,
            core_specification=core_specification,
            thread_specification=thread_specification,
            cpu_binding=cpu_binding,
            cpu_binding_flags=cpu_binding_flags,
            cpu_frequency=cpu_frequency,
            cpus_per_tres=cpus_per_tres,
            crontab=crontab,
            deadline=deadline,
            delay_boot=delay_boot,
            dependency=dependency,
            end_time=end_time,
            environment=environment,
            rlimits=rlimits,
            excluded_nodes=excluded_nodes,
            extra=extra,
            constraints=constraints,
            group_id=group_id,
            hetjob_group=hetjob_group,
            immediate=immediate,
            job_id=job_id,
            kill_on_node_fail=kill_on_node_fail,
            licenses=licenses,
            mail_type=mail_type,
            mail_user=mail_user,
            mcs_label=mcs_label,
            memory_binding=memory_binding,
            memory_binding_type=memory_binding_type,
            memory_per_tres=memory_per_tres,
            name=name,
            network=network,
            nice=nice,
            tasks=tasks,
            oom_kill_step=oom_kill_step,
            open_mode=open_mode,
            reserve_ports=reserve_ports,
            overcommit=overcommit,
            partition=partition,
            distribution_plane_size=distribution_plane_size,
            power_flags=power_flags,
            prefer=prefer,
            hold=hold,
            priority=priority,
            profile=profile,
            qos=qos,
            reboot=reboot,
            required_nodes=required_nodes,
            requeue=requeue,
            reservation=reservation,
            script=script,
            shared=shared,
            site_factor=site_factor,
            spank_environment=spank_environment,
            distribution=distribution,
            time_limit=time_limit,
            time_minimum=time_minimum,
            tres_bind=tres_bind,
            tres_freq=tres_freq,
            tres_per_job=tres_per_job,
            tres_per_node=tres_per_node,
            tres_per_socket=tres_per_socket,
            tres_per_task=tres_per_task,
            user_id=user_id,
            wait_all_nodes=wait_all_nodes,
            kill_warning_flags=kill_warning_flags,
            kill_warning_signal=kill_warning_signal,
            kill_warning_delay=kill_warning_delay,
            current_working_directory=current_working_directory,
            cpus_per_task=cpus_per_task,
            minimum_cpus=minimum_cpus,
            maximum_cpus=maximum_cpus,
            nodes=nodes,
            minimum_nodes=minimum_nodes,
            maximum_nodes=maximum_nodes,
            minimum_boards_per_node=minimum_boards_per_node,
            minimum_sockets_per_board=minimum_sockets_per_board,
            sockets_per_node=sockets_per_node,
            threads_per_core=threads_per_core,
            tasks_per_node=tasks_per_node,
            tasks_per_socket=tasks_per_socket,
            tasks_per_core=tasks_per_core,
            tasks_per_board=tasks_per_board,
            ntasks_per_tres=ntasks_per_tres,
            minimum_cpus_per_node=minimum_cpus_per_node,
            memory_per_cpu=memory_per_cpu,
            memory_per_node=memory_per_node,
            temporary_disk_per_node=temporary_disk_per_node,
            selinux_context=selinux_context,
            required_switches=required_switches,
            segment_size=segment_size,
            standard_error=standard_error,
            standard_input=standard_input,
            standard_output=standard_output,
            wait_for_switch=wait_for_switch,
            wckey=wckey,
            x11=x11,
            x11_magic_cookie=x11_magic_cookie,
            x11_target_host=x11_target_host,
            x11_target_port=x11_target_port,
        )

        v0043_job_desc_msg.additional_properties = d
        return v0043_job_desc_msg

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
