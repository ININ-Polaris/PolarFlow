from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_bf_exit_fields import V0043BfExitFields
    from ..models.v0043_schedule_exit_fields import V0043ScheduleExitFields
    from ..models.v0043_stats_msg_rpc_dump import V0043StatsMsgRpcDump
    from ..models.v0043_stats_msg_rpc_queue import V0043StatsMsgRpcQueue
    from ..models.v0043_stats_msg_rpc_type import V0043StatsMsgRpcType
    from ..models.v0043_stats_msg_rpc_user import V0043StatsMsgRpcUser
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043StatsMsg")


@_attrs_define
class V0043StatsMsg:
    """
    Attributes:
        parts_packed (Union[Unset, int]): Zero if only RPC statistic included
        req_time (Union[Unset, V0043Uint64NoValStruct]):
        req_time_start (Union[Unset, V0043Uint64NoValStruct]):
        server_thread_count (Union[Unset, int]): Number of current active slurmctld threads
        agent_queue_size (Union[Unset, int]): Number of enqueued outgoing RPC requests in an internal retry list
        agent_count (Union[Unset, int]): Number of agent threads
        agent_thread_count (Union[Unset, int]): Total number of active threads created by all agent threads
        dbd_agent_queue_size (Union[Unset, int]): Number of messages for SlurmDBD that are queued
        gettimeofday_latency (Union[Unset, int]): Latency of 1000 calls to the gettimeofday() syscall in microseconds,
            as measured at controller startup
        schedule_cycle_max (Union[Unset, int]): Max time of any scheduling cycle in microseconds since last reset
        schedule_cycle_last (Union[Unset, int]): Time in microseconds for last scheduling cycle
        schedule_cycle_sum (Union[Unset, int]): Total run time in microseconds for all scheduling cycles since last
            reset
        schedule_cycle_total (Union[Unset, int]): Number of scheduling cycles since last reset
        schedule_cycle_mean (Union[Unset, int]): Mean time in microseconds for all scheduling cycles since last reset
        schedule_cycle_mean_depth (Union[Unset, int]): Mean of the number of jobs processed in a scheduling cycle
        schedule_cycle_per_minute (Union[Unset, int]): Number of scheduling executions per minute
        schedule_cycle_depth (Union[Unset, int]): Total number of jobs processed in scheduling cycles
        schedule_exit (Union[Unset, V0043ScheduleExitFields]):
        schedule_queue_length (Union[Unset, int]): Number of jobs pending in queue
        jobs_submitted (Union[Unset, int]): Number of jobs submitted since last reset
        jobs_started (Union[Unset, int]): Number of jobs started since last reset
        jobs_completed (Union[Unset, int]): Number of jobs completed since last reset
        jobs_canceled (Union[Unset, int]): Number of jobs canceled since the last reset
        jobs_failed (Union[Unset, int]): Number of jobs failed due to slurmd or other internal issues since last reset
        jobs_pending (Union[Unset, int]): Number of jobs pending at the time of listed in job_state_ts
        jobs_running (Union[Unset, int]): Number of jobs running at the time of listed in job_state_ts
        job_states_ts (Union[Unset, V0043Uint64NoValStruct]):
        bf_backfilled_jobs (Union[Unset, int]): Number of jobs started through backfilling since last slurm start
        bf_last_backfilled_jobs (Union[Unset, int]): Number of jobs started through backfilling since last reset
        bf_backfilled_het_jobs (Union[Unset, int]): Number of heterogeneous job components started through backfilling
            since last Slurm start
        bf_cycle_counter (Union[Unset, int]): Number of backfill scheduling cycles since last reset
        bf_cycle_mean (Union[Unset, int]): Mean time in microseconds of backfilling scheduling cycles since last reset
        bf_depth_mean (Union[Unset, int]): Mean number of eligible to run jobs processed during all backfilling
            scheduling cycles since last reset
        bf_depth_mean_try (Union[Unset, int]): The subset of Depth Mean that the backfill scheduler attempted to
            schedule
        bf_cycle_sum (Union[Unset, int]): Total time in microseconds of backfilling scheduling cycles since last reset
        bf_cycle_last (Union[Unset, int]): Execution time in microseconds of last backfill scheduling cycle
        bf_cycle_max (Union[Unset, int]): Execution time in microseconds of longest backfill scheduling cycle
        bf_exit (Union[Unset, V0043BfExitFields]):
        bf_last_depth (Union[Unset, int]): Number of processed jobs during last backfilling scheduling cycle
        bf_last_depth_try (Union[Unset, int]): Number of processed jobs during last backfilling scheduling cycle that
            had a chance to start using available resources
        bf_depth_sum (Union[Unset, int]): Total number of jobs processed during all backfilling scheduling cycles since
            last reset
        bf_depth_try_sum (Union[Unset, int]): Subset of bf_depth_sum that the backfill scheduler attempted to schedule
        bf_queue_len (Union[Unset, int]): Number of jobs pending to be processed by backfilling algorithm
        bf_queue_len_mean (Union[Unset, int]): Mean number of jobs pending to be processed by backfilling algorithm
        bf_queue_len_sum (Union[Unset, int]): Total number of jobs pending to be processed by backfilling algorithm
            since last reset
        bf_table_size (Union[Unset, int]): Number of different time slots tested by the backfill scheduler in its last
            iteration
        bf_table_size_sum (Union[Unset, int]): Total number of different time slots tested by the backfill scheduler
        bf_table_size_mean (Union[Unset, int]): Mean number of different time slots tested by the backfill scheduler
        bf_when_last_cycle (Union[Unset, V0043Uint64NoValStruct]):
        bf_active (Union[Unset, bool]): Backfill scheduler currently running
        rpcs_by_message_type (Union[Unset, list['V0043StatsMsgRpcType']]): RPCs by type
        rpcs_by_user (Union[Unset, list['V0043StatsMsgRpcUser']]): RPCs by user
        pending_rpcs (Union[Unset, list['V0043StatsMsgRpcQueue']]): Pending RPCs
        pending_rpcs_by_hostlist (Union[Unset, list['V0043StatsMsgRpcDump']]): Pending RPCs by hostlist
    """

    parts_packed: Union[Unset, int] = UNSET
    req_time: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    req_time_start: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    server_thread_count: Union[Unset, int] = UNSET
    agent_queue_size: Union[Unset, int] = UNSET
    agent_count: Union[Unset, int] = UNSET
    agent_thread_count: Union[Unset, int] = UNSET
    dbd_agent_queue_size: Union[Unset, int] = UNSET
    gettimeofday_latency: Union[Unset, int] = UNSET
    schedule_cycle_max: Union[Unset, int] = UNSET
    schedule_cycle_last: Union[Unset, int] = UNSET
    schedule_cycle_sum: Union[Unset, int] = UNSET
    schedule_cycle_total: Union[Unset, int] = UNSET
    schedule_cycle_mean: Union[Unset, int] = UNSET
    schedule_cycle_mean_depth: Union[Unset, int] = UNSET
    schedule_cycle_per_minute: Union[Unset, int] = UNSET
    schedule_cycle_depth: Union[Unset, int] = UNSET
    schedule_exit: Union[Unset, "V0043ScheduleExitFields"] = UNSET
    schedule_queue_length: Union[Unset, int] = UNSET
    jobs_submitted: Union[Unset, int] = UNSET
    jobs_started: Union[Unset, int] = UNSET
    jobs_completed: Union[Unset, int] = UNSET
    jobs_canceled: Union[Unset, int] = UNSET
    jobs_failed: Union[Unset, int] = UNSET
    jobs_pending: Union[Unset, int] = UNSET
    jobs_running: Union[Unset, int] = UNSET
    job_states_ts: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    bf_backfilled_jobs: Union[Unset, int] = UNSET
    bf_last_backfilled_jobs: Union[Unset, int] = UNSET
    bf_backfilled_het_jobs: Union[Unset, int] = UNSET
    bf_cycle_counter: Union[Unset, int] = UNSET
    bf_cycle_mean: Union[Unset, int] = UNSET
    bf_depth_mean: Union[Unset, int] = UNSET
    bf_depth_mean_try: Union[Unset, int] = UNSET
    bf_cycle_sum: Union[Unset, int] = UNSET
    bf_cycle_last: Union[Unset, int] = UNSET
    bf_cycle_max: Union[Unset, int] = UNSET
    bf_exit: Union[Unset, "V0043BfExitFields"] = UNSET
    bf_last_depth: Union[Unset, int] = UNSET
    bf_last_depth_try: Union[Unset, int] = UNSET
    bf_depth_sum: Union[Unset, int] = UNSET
    bf_depth_try_sum: Union[Unset, int] = UNSET
    bf_queue_len: Union[Unset, int] = UNSET
    bf_queue_len_mean: Union[Unset, int] = UNSET
    bf_queue_len_sum: Union[Unset, int] = UNSET
    bf_table_size: Union[Unset, int] = UNSET
    bf_table_size_sum: Union[Unset, int] = UNSET
    bf_table_size_mean: Union[Unset, int] = UNSET
    bf_when_last_cycle: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    bf_active: Union[Unset, bool] = UNSET
    rpcs_by_message_type: Union[Unset, list["V0043StatsMsgRpcType"]] = UNSET
    rpcs_by_user: Union[Unset, list["V0043StatsMsgRpcUser"]] = UNSET
    pending_rpcs: Union[Unset, list["V0043StatsMsgRpcQueue"]] = UNSET
    pending_rpcs_by_hostlist: Union[Unset, list["V0043StatsMsgRpcDump"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        parts_packed = self.parts_packed

        req_time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.req_time, Unset):
            req_time = self.req_time.to_dict()

        req_time_start: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.req_time_start, Unset):
            req_time_start = self.req_time_start.to_dict()

        server_thread_count = self.server_thread_count

        agent_queue_size = self.agent_queue_size

        agent_count = self.agent_count

        agent_thread_count = self.agent_thread_count

        dbd_agent_queue_size = self.dbd_agent_queue_size

        gettimeofday_latency = self.gettimeofday_latency

        schedule_cycle_max = self.schedule_cycle_max

        schedule_cycle_last = self.schedule_cycle_last

        schedule_cycle_sum = self.schedule_cycle_sum

        schedule_cycle_total = self.schedule_cycle_total

        schedule_cycle_mean = self.schedule_cycle_mean

        schedule_cycle_mean_depth = self.schedule_cycle_mean_depth

        schedule_cycle_per_minute = self.schedule_cycle_per_minute

        schedule_cycle_depth = self.schedule_cycle_depth

        schedule_exit: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.schedule_exit, Unset):
            schedule_exit = self.schedule_exit.to_dict()

        schedule_queue_length = self.schedule_queue_length

        jobs_submitted = self.jobs_submitted

        jobs_started = self.jobs_started

        jobs_completed = self.jobs_completed

        jobs_canceled = self.jobs_canceled

        jobs_failed = self.jobs_failed

        jobs_pending = self.jobs_pending

        jobs_running = self.jobs_running

        job_states_ts: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.job_states_ts, Unset):
            job_states_ts = self.job_states_ts.to_dict()

        bf_backfilled_jobs = self.bf_backfilled_jobs

        bf_last_backfilled_jobs = self.bf_last_backfilled_jobs

        bf_backfilled_het_jobs = self.bf_backfilled_het_jobs

        bf_cycle_counter = self.bf_cycle_counter

        bf_cycle_mean = self.bf_cycle_mean

        bf_depth_mean = self.bf_depth_mean

        bf_depth_mean_try = self.bf_depth_mean_try

        bf_cycle_sum = self.bf_cycle_sum

        bf_cycle_last = self.bf_cycle_last

        bf_cycle_max = self.bf_cycle_max

        bf_exit: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.bf_exit, Unset):
            bf_exit = self.bf_exit.to_dict()

        bf_last_depth = self.bf_last_depth

        bf_last_depth_try = self.bf_last_depth_try

        bf_depth_sum = self.bf_depth_sum

        bf_depth_try_sum = self.bf_depth_try_sum

        bf_queue_len = self.bf_queue_len

        bf_queue_len_mean = self.bf_queue_len_mean

        bf_queue_len_sum = self.bf_queue_len_sum

        bf_table_size = self.bf_table_size

        bf_table_size_sum = self.bf_table_size_sum

        bf_table_size_mean = self.bf_table_size_mean

        bf_when_last_cycle: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.bf_when_last_cycle, Unset):
            bf_when_last_cycle = self.bf_when_last_cycle.to_dict()

        bf_active = self.bf_active

        rpcs_by_message_type: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.rpcs_by_message_type, Unset):
            rpcs_by_message_type = []
            for componentsschemasv0_0_43_stats_msg_rpcs_by_type_item_data in self.rpcs_by_message_type:
                componentsschemasv0_0_43_stats_msg_rpcs_by_type_item = (
                    componentsschemasv0_0_43_stats_msg_rpcs_by_type_item_data.to_dict()
                )
                rpcs_by_message_type.append(componentsschemasv0_0_43_stats_msg_rpcs_by_type_item)

        rpcs_by_user: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.rpcs_by_user, Unset):
            rpcs_by_user = []
            for componentsschemasv0_0_43_stats_msg_rpcs_by_user_item_data in self.rpcs_by_user:
                componentsschemasv0_0_43_stats_msg_rpcs_by_user_item = (
                    componentsschemasv0_0_43_stats_msg_rpcs_by_user_item_data.to_dict()
                )
                rpcs_by_user.append(componentsschemasv0_0_43_stats_msg_rpcs_by_user_item)

        pending_rpcs: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.pending_rpcs, Unset):
            pending_rpcs = []
            for componentsschemasv0_0_43_stats_msg_rpcs_queue_item_data in self.pending_rpcs:
                componentsschemasv0_0_43_stats_msg_rpcs_queue_item = (
                    componentsschemasv0_0_43_stats_msg_rpcs_queue_item_data.to_dict()
                )
                pending_rpcs.append(componentsschemasv0_0_43_stats_msg_rpcs_queue_item)

        pending_rpcs_by_hostlist: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.pending_rpcs_by_hostlist, Unset):
            pending_rpcs_by_hostlist = []
            for componentsschemasv0_0_43_stats_msg_rpcs_dump_item_data in self.pending_rpcs_by_hostlist:
                componentsschemasv0_0_43_stats_msg_rpcs_dump_item = (
                    componentsschemasv0_0_43_stats_msg_rpcs_dump_item_data.to_dict()
                )
                pending_rpcs_by_hostlist.append(componentsschemasv0_0_43_stats_msg_rpcs_dump_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if parts_packed is not UNSET:
            field_dict["parts_packed"] = parts_packed
        if req_time is not UNSET:
            field_dict["req_time"] = req_time
        if req_time_start is not UNSET:
            field_dict["req_time_start"] = req_time_start
        if server_thread_count is not UNSET:
            field_dict["server_thread_count"] = server_thread_count
        if agent_queue_size is not UNSET:
            field_dict["agent_queue_size"] = agent_queue_size
        if agent_count is not UNSET:
            field_dict["agent_count"] = agent_count
        if agent_thread_count is not UNSET:
            field_dict["agent_thread_count"] = agent_thread_count
        if dbd_agent_queue_size is not UNSET:
            field_dict["dbd_agent_queue_size"] = dbd_agent_queue_size
        if gettimeofday_latency is not UNSET:
            field_dict["gettimeofday_latency"] = gettimeofday_latency
        if schedule_cycle_max is not UNSET:
            field_dict["schedule_cycle_max"] = schedule_cycle_max
        if schedule_cycle_last is not UNSET:
            field_dict["schedule_cycle_last"] = schedule_cycle_last
        if schedule_cycle_sum is not UNSET:
            field_dict["schedule_cycle_sum"] = schedule_cycle_sum
        if schedule_cycle_total is not UNSET:
            field_dict["schedule_cycle_total"] = schedule_cycle_total
        if schedule_cycle_mean is not UNSET:
            field_dict["schedule_cycle_mean"] = schedule_cycle_mean
        if schedule_cycle_mean_depth is not UNSET:
            field_dict["schedule_cycle_mean_depth"] = schedule_cycle_mean_depth
        if schedule_cycle_per_minute is not UNSET:
            field_dict["schedule_cycle_per_minute"] = schedule_cycle_per_minute
        if schedule_cycle_depth is not UNSET:
            field_dict["schedule_cycle_depth"] = schedule_cycle_depth
        if schedule_exit is not UNSET:
            field_dict["schedule_exit"] = schedule_exit
        if schedule_queue_length is not UNSET:
            field_dict["schedule_queue_length"] = schedule_queue_length
        if jobs_submitted is not UNSET:
            field_dict["jobs_submitted"] = jobs_submitted
        if jobs_started is not UNSET:
            field_dict["jobs_started"] = jobs_started
        if jobs_completed is not UNSET:
            field_dict["jobs_completed"] = jobs_completed
        if jobs_canceled is not UNSET:
            field_dict["jobs_canceled"] = jobs_canceled
        if jobs_failed is not UNSET:
            field_dict["jobs_failed"] = jobs_failed
        if jobs_pending is not UNSET:
            field_dict["jobs_pending"] = jobs_pending
        if jobs_running is not UNSET:
            field_dict["jobs_running"] = jobs_running
        if job_states_ts is not UNSET:
            field_dict["job_states_ts"] = job_states_ts
        if bf_backfilled_jobs is not UNSET:
            field_dict["bf_backfilled_jobs"] = bf_backfilled_jobs
        if bf_last_backfilled_jobs is not UNSET:
            field_dict["bf_last_backfilled_jobs"] = bf_last_backfilled_jobs
        if bf_backfilled_het_jobs is not UNSET:
            field_dict["bf_backfilled_het_jobs"] = bf_backfilled_het_jobs
        if bf_cycle_counter is not UNSET:
            field_dict["bf_cycle_counter"] = bf_cycle_counter
        if bf_cycle_mean is not UNSET:
            field_dict["bf_cycle_mean"] = bf_cycle_mean
        if bf_depth_mean is not UNSET:
            field_dict["bf_depth_mean"] = bf_depth_mean
        if bf_depth_mean_try is not UNSET:
            field_dict["bf_depth_mean_try"] = bf_depth_mean_try
        if bf_cycle_sum is not UNSET:
            field_dict["bf_cycle_sum"] = bf_cycle_sum
        if bf_cycle_last is not UNSET:
            field_dict["bf_cycle_last"] = bf_cycle_last
        if bf_cycle_max is not UNSET:
            field_dict["bf_cycle_max"] = bf_cycle_max
        if bf_exit is not UNSET:
            field_dict["bf_exit"] = bf_exit
        if bf_last_depth is not UNSET:
            field_dict["bf_last_depth"] = bf_last_depth
        if bf_last_depth_try is not UNSET:
            field_dict["bf_last_depth_try"] = bf_last_depth_try
        if bf_depth_sum is not UNSET:
            field_dict["bf_depth_sum"] = bf_depth_sum
        if bf_depth_try_sum is not UNSET:
            field_dict["bf_depth_try_sum"] = bf_depth_try_sum
        if bf_queue_len is not UNSET:
            field_dict["bf_queue_len"] = bf_queue_len
        if bf_queue_len_mean is not UNSET:
            field_dict["bf_queue_len_mean"] = bf_queue_len_mean
        if bf_queue_len_sum is not UNSET:
            field_dict["bf_queue_len_sum"] = bf_queue_len_sum
        if bf_table_size is not UNSET:
            field_dict["bf_table_size"] = bf_table_size
        if bf_table_size_sum is not UNSET:
            field_dict["bf_table_size_sum"] = bf_table_size_sum
        if bf_table_size_mean is not UNSET:
            field_dict["bf_table_size_mean"] = bf_table_size_mean
        if bf_when_last_cycle is not UNSET:
            field_dict["bf_when_last_cycle"] = bf_when_last_cycle
        if bf_active is not UNSET:
            field_dict["bf_active"] = bf_active
        if rpcs_by_message_type is not UNSET:
            field_dict["rpcs_by_message_type"] = rpcs_by_message_type
        if rpcs_by_user is not UNSET:
            field_dict["rpcs_by_user"] = rpcs_by_user
        if pending_rpcs is not UNSET:
            field_dict["pending_rpcs"] = pending_rpcs
        if pending_rpcs_by_hostlist is not UNSET:
            field_dict["pending_rpcs_by_hostlist"] = pending_rpcs_by_hostlist

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_bf_exit_fields import V0043BfExitFields
        from ..models.v0043_schedule_exit_fields import V0043ScheduleExitFields
        from ..models.v0043_stats_msg_rpc_dump import V0043StatsMsgRpcDump
        from ..models.v0043_stats_msg_rpc_queue import V0043StatsMsgRpcQueue
        from ..models.v0043_stats_msg_rpc_type import V0043StatsMsgRpcType
        from ..models.v0043_stats_msg_rpc_user import V0043StatsMsgRpcUser
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        parts_packed = d.pop("parts_packed", UNSET)

        _req_time = d.pop("req_time", UNSET)
        req_time: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_req_time, Unset):
            req_time = UNSET
        else:
            req_time = V0043Uint64NoValStruct.from_dict(_req_time)

        _req_time_start = d.pop("req_time_start", UNSET)
        req_time_start: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_req_time_start, Unset):
            req_time_start = UNSET
        else:
            req_time_start = V0043Uint64NoValStruct.from_dict(_req_time_start)

        server_thread_count = d.pop("server_thread_count", UNSET)

        agent_queue_size = d.pop("agent_queue_size", UNSET)

        agent_count = d.pop("agent_count", UNSET)

        agent_thread_count = d.pop("agent_thread_count", UNSET)

        dbd_agent_queue_size = d.pop("dbd_agent_queue_size", UNSET)

        gettimeofday_latency = d.pop("gettimeofday_latency", UNSET)

        schedule_cycle_max = d.pop("schedule_cycle_max", UNSET)

        schedule_cycle_last = d.pop("schedule_cycle_last", UNSET)

        schedule_cycle_sum = d.pop("schedule_cycle_sum", UNSET)

        schedule_cycle_total = d.pop("schedule_cycle_total", UNSET)

        schedule_cycle_mean = d.pop("schedule_cycle_mean", UNSET)

        schedule_cycle_mean_depth = d.pop("schedule_cycle_mean_depth", UNSET)

        schedule_cycle_per_minute = d.pop("schedule_cycle_per_minute", UNSET)

        schedule_cycle_depth = d.pop("schedule_cycle_depth", UNSET)

        _schedule_exit = d.pop("schedule_exit", UNSET)
        schedule_exit: Union[Unset, V0043ScheduleExitFields]
        if isinstance(_schedule_exit, Unset):
            schedule_exit = UNSET
        else:
            schedule_exit = V0043ScheduleExitFields.from_dict(_schedule_exit)

        schedule_queue_length = d.pop("schedule_queue_length", UNSET)

        jobs_submitted = d.pop("jobs_submitted", UNSET)

        jobs_started = d.pop("jobs_started", UNSET)

        jobs_completed = d.pop("jobs_completed", UNSET)

        jobs_canceled = d.pop("jobs_canceled", UNSET)

        jobs_failed = d.pop("jobs_failed", UNSET)

        jobs_pending = d.pop("jobs_pending", UNSET)

        jobs_running = d.pop("jobs_running", UNSET)

        _job_states_ts = d.pop("job_states_ts", UNSET)
        job_states_ts: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_job_states_ts, Unset):
            job_states_ts = UNSET
        else:
            job_states_ts = V0043Uint64NoValStruct.from_dict(_job_states_ts)

        bf_backfilled_jobs = d.pop("bf_backfilled_jobs", UNSET)

        bf_last_backfilled_jobs = d.pop("bf_last_backfilled_jobs", UNSET)

        bf_backfilled_het_jobs = d.pop("bf_backfilled_het_jobs", UNSET)

        bf_cycle_counter = d.pop("bf_cycle_counter", UNSET)

        bf_cycle_mean = d.pop("bf_cycle_mean", UNSET)

        bf_depth_mean = d.pop("bf_depth_mean", UNSET)

        bf_depth_mean_try = d.pop("bf_depth_mean_try", UNSET)

        bf_cycle_sum = d.pop("bf_cycle_sum", UNSET)

        bf_cycle_last = d.pop("bf_cycle_last", UNSET)

        bf_cycle_max = d.pop("bf_cycle_max", UNSET)

        _bf_exit = d.pop("bf_exit", UNSET)
        bf_exit: Union[Unset, V0043BfExitFields]
        if isinstance(_bf_exit, Unset):
            bf_exit = UNSET
        else:
            bf_exit = V0043BfExitFields.from_dict(_bf_exit)

        bf_last_depth = d.pop("bf_last_depth", UNSET)

        bf_last_depth_try = d.pop("bf_last_depth_try", UNSET)

        bf_depth_sum = d.pop("bf_depth_sum", UNSET)

        bf_depth_try_sum = d.pop("bf_depth_try_sum", UNSET)

        bf_queue_len = d.pop("bf_queue_len", UNSET)

        bf_queue_len_mean = d.pop("bf_queue_len_mean", UNSET)

        bf_queue_len_sum = d.pop("bf_queue_len_sum", UNSET)

        bf_table_size = d.pop("bf_table_size", UNSET)

        bf_table_size_sum = d.pop("bf_table_size_sum", UNSET)

        bf_table_size_mean = d.pop("bf_table_size_mean", UNSET)

        _bf_when_last_cycle = d.pop("bf_when_last_cycle", UNSET)
        bf_when_last_cycle: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_bf_when_last_cycle, Unset):
            bf_when_last_cycle = UNSET
        else:
            bf_when_last_cycle = V0043Uint64NoValStruct.from_dict(_bf_when_last_cycle)

        bf_active = d.pop("bf_active", UNSET)

        rpcs_by_message_type = []
        _rpcs_by_message_type = d.pop("rpcs_by_message_type", UNSET)
        for componentsschemasv0_0_43_stats_msg_rpcs_by_type_item_data in _rpcs_by_message_type or []:
            componentsschemasv0_0_43_stats_msg_rpcs_by_type_item = V0043StatsMsgRpcType.from_dict(
                componentsschemasv0_0_43_stats_msg_rpcs_by_type_item_data
            )

            rpcs_by_message_type.append(componentsschemasv0_0_43_stats_msg_rpcs_by_type_item)

        rpcs_by_user = []
        _rpcs_by_user = d.pop("rpcs_by_user", UNSET)
        for componentsschemasv0_0_43_stats_msg_rpcs_by_user_item_data in _rpcs_by_user or []:
            componentsschemasv0_0_43_stats_msg_rpcs_by_user_item = V0043StatsMsgRpcUser.from_dict(
                componentsschemasv0_0_43_stats_msg_rpcs_by_user_item_data
            )

            rpcs_by_user.append(componentsschemasv0_0_43_stats_msg_rpcs_by_user_item)

        pending_rpcs = []
        _pending_rpcs = d.pop("pending_rpcs", UNSET)
        for componentsschemasv0_0_43_stats_msg_rpcs_queue_item_data in _pending_rpcs or []:
            componentsschemasv0_0_43_stats_msg_rpcs_queue_item = V0043StatsMsgRpcQueue.from_dict(
                componentsschemasv0_0_43_stats_msg_rpcs_queue_item_data
            )

            pending_rpcs.append(componentsschemasv0_0_43_stats_msg_rpcs_queue_item)

        pending_rpcs_by_hostlist = []
        _pending_rpcs_by_hostlist = d.pop("pending_rpcs_by_hostlist", UNSET)
        for componentsschemasv0_0_43_stats_msg_rpcs_dump_item_data in _pending_rpcs_by_hostlist or []:
            componentsschemasv0_0_43_stats_msg_rpcs_dump_item = V0043StatsMsgRpcDump.from_dict(
                componentsschemasv0_0_43_stats_msg_rpcs_dump_item_data
            )

            pending_rpcs_by_hostlist.append(componentsschemasv0_0_43_stats_msg_rpcs_dump_item)

        v0043_stats_msg = cls(
            parts_packed=parts_packed,
            req_time=req_time,
            req_time_start=req_time_start,
            server_thread_count=server_thread_count,
            agent_queue_size=agent_queue_size,
            agent_count=agent_count,
            agent_thread_count=agent_thread_count,
            dbd_agent_queue_size=dbd_agent_queue_size,
            gettimeofday_latency=gettimeofday_latency,
            schedule_cycle_max=schedule_cycle_max,
            schedule_cycle_last=schedule_cycle_last,
            schedule_cycle_sum=schedule_cycle_sum,
            schedule_cycle_total=schedule_cycle_total,
            schedule_cycle_mean=schedule_cycle_mean,
            schedule_cycle_mean_depth=schedule_cycle_mean_depth,
            schedule_cycle_per_minute=schedule_cycle_per_minute,
            schedule_cycle_depth=schedule_cycle_depth,
            schedule_exit=schedule_exit,
            schedule_queue_length=schedule_queue_length,
            jobs_submitted=jobs_submitted,
            jobs_started=jobs_started,
            jobs_completed=jobs_completed,
            jobs_canceled=jobs_canceled,
            jobs_failed=jobs_failed,
            jobs_pending=jobs_pending,
            jobs_running=jobs_running,
            job_states_ts=job_states_ts,
            bf_backfilled_jobs=bf_backfilled_jobs,
            bf_last_backfilled_jobs=bf_last_backfilled_jobs,
            bf_backfilled_het_jobs=bf_backfilled_het_jobs,
            bf_cycle_counter=bf_cycle_counter,
            bf_cycle_mean=bf_cycle_mean,
            bf_depth_mean=bf_depth_mean,
            bf_depth_mean_try=bf_depth_mean_try,
            bf_cycle_sum=bf_cycle_sum,
            bf_cycle_last=bf_cycle_last,
            bf_cycle_max=bf_cycle_max,
            bf_exit=bf_exit,
            bf_last_depth=bf_last_depth,
            bf_last_depth_try=bf_last_depth_try,
            bf_depth_sum=bf_depth_sum,
            bf_depth_try_sum=bf_depth_try_sum,
            bf_queue_len=bf_queue_len,
            bf_queue_len_mean=bf_queue_len_mean,
            bf_queue_len_sum=bf_queue_len_sum,
            bf_table_size=bf_table_size,
            bf_table_size_sum=bf_table_size_sum,
            bf_table_size_mean=bf_table_size_mean,
            bf_when_last_cycle=bf_when_last_cycle,
            bf_active=bf_active,
            rpcs_by_message_type=rpcs_by_message_type,
            rpcs_by_user=rpcs_by_user,
            pending_rpcs=pending_rpcs,
            pending_rpcs_by_hostlist=pending_rpcs_by_hostlist,
        )

        v0043_stats_msg.additional_properties = d
        return v0043_stats_msg

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
