"""Contains all the data models used in inputs/outputs"""

from .slurm_v0043_delete_job_flags import SlurmV0043DeleteJobFlags
from .slurm_v0043_get_job_flags import SlurmV0043GetJobFlags
from .slurm_v0043_get_jobs_flags import SlurmV0043GetJobsFlags
from .slurm_v0043_get_node_flags import SlurmV0043GetNodeFlags
from .slurm_v0043_get_nodes_flags import SlurmV0043GetNodesFlags
from .slurm_v0043_get_partition_flags import SlurmV0043GetPartitionFlags
from .slurm_v0043_get_partitions_flags import SlurmV0043GetPartitionsFlags
from .slurmdb_v0043_delete_cluster_classification import SlurmdbV0043DeleteClusterClassification
from .slurmdb_v0043_delete_cluster_flags import SlurmdbV0043DeleteClusterFlags
from .slurmdb_v0043_get_cluster_classification import SlurmdbV0043GetClusterClassification
from .slurmdb_v0043_get_cluster_flags import SlurmdbV0043GetClusterFlags
from .slurmdb_v0043_get_qos_preempt_mode import SlurmdbV0043GetQosPreemptMode
from .slurmdb_v0043_get_users_admin_level import SlurmdbV0043GetUsersAdminLevel
from .slurmdb_v0043_post_qos_preempt_mode import SlurmdbV0043PostQosPreemptMode
from .slurmdb_v0043_post_users_association_flags import SlurmdbV0043PostUsersAssociationFlags
from .v0043_account import V0043Account
from .v0043_account_flags_item import V0043AccountFlagsItem
from .v0043_account_short import V0043AccountShort
from .v0043_accounting import V0043Accounting
from .v0043_accounting_allocated import V0043AccountingAllocated
from .v0043_accounts_add_cond import V0043AccountsAddCond
from .v0043_acct_gather_energy import V0043AcctGatherEnergy
from .v0043_assoc import V0043Assoc
from .v0043_assoc_default import V0043AssocDefault
from .v0043_assoc_flags_item import V0043AssocFlagsItem
from .v0043_assoc_max import V0043AssocMax
from .v0043_assoc_max_jobs import V0043AssocMaxJobs
from .v0043_assoc_max_jobs_per import V0043AssocMaxJobsPer
from .v0043_assoc_max_per import V0043AssocMaxPer
from .v0043_assoc_max_per_account import V0043AssocMaxPerAccount
from .v0043_assoc_max_tres import V0043AssocMaxTres
from .v0043_assoc_max_tres_group import V0043AssocMaxTresGroup
from .v0043_assoc_max_tres_minutes import V0043AssocMaxTresMinutes
from .v0043_assoc_max_tres_minutes_per import V0043AssocMaxTresMinutesPer
from .v0043_assoc_max_tres_per import V0043AssocMaxTresPer
from .v0043_assoc_min import V0043AssocMin
from .v0043_assoc_rec_set import V0043AssocRecSet
from .v0043_assoc_shares_obj_wrap import V0043AssocSharesObjWrap
from .v0043_assoc_shares_obj_wrap_fairshare import V0043AssocSharesObjWrapFairshare
from .v0043_assoc_shares_obj_wrap_tres import V0043AssocSharesObjWrapTres
from .v0043_assoc_shares_obj_wrap_type_item import V0043AssocSharesObjWrapTypeItem
from .v0043_assoc_short import V0043AssocShort
from .v0043_bf_exit_fields import V0043BfExitFields
from .v0043_cluster_rec import V0043ClusterRec
from .v0043_cluster_rec_associations import V0043ClusterRecAssociations
from .v0043_cluster_rec_controller import V0043ClusterRecController
from .v0043_cluster_rec_flags_item import V0043ClusterRecFlagsItem
from .v0043_controller_ping import V0043ControllerPing
from .v0043_coord import V0043Coord
from .v0043_cron_entry import V0043CronEntry
from .v0043_cron_entry_flags_item import V0043CronEntryFlagsItem
from .v0043_cron_entry_line import V0043CronEntryLine
from .v0043_float_64_no_val_struct import V0043Float64NoValStruct
from .v0043_instance import V0043Instance
from .v0043_instance_time import V0043InstanceTime
from .v0043_job import V0043Job
from .v0043_job_alloc_req import V0043JobAllocReq
from .v0043_job_array import V0043JobArray
from .v0043_job_array_limits import V0043JobArrayLimits
from .v0043_job_array_limits_max import V0043JobArrayLimitsMax
from .v0043_job_array_limits_max_running import V0043JobArrayLimitsMaxRunning
from .v0043_job_array_response_msg_entry import V0043JobArrayResponseMsgEntry
from .v0043_job_comment import V0043JobComment
from .v0043_job_desc_msg import V0043JobDescMsg
from .v0043_job_desc_msg_cpu_binding_flags_item import V0043JobDescMsgCpuBindingFlagsItem
from .v0043_job_desc_msg_flags_item import V0043JobDescMsgFlagsItem
from .v0043_job_desc_msg_kill_warning_flags_item import V0043JobDescMsgKillWarningFlagsItem
from .v0043_job_desc_msg_mail_type_item import V0043JobDescMsgMailTypeItem
from .v0043_job_desc_msg_memory_binding_type_item import V0043JobDescMsgMemoryBindingTypeItem
from .v0043_job_desc_msg_open_mode_item import V0043JobDescMsgOpenModeItem
from .v0043_job_desc_msg_profile_item import V0043JobDescMsgProfileItem
from .v0043_job_desc_msg_rlimits import V0043JobDescMsgRlimits
from .v0043_job_desc_msg_shared_item import V0043JobDescMsgSharedItem
from .v0043_job_desc_msg_x11_item import V0043JobDescMsgX11Item
from .v0043_job_flags_item import V0043JobFlagsItem
from .v0043_job_het import V0043JobHet
from .v0043_job_info import V0043JobInfo
from .v0043_job_info_flags_item import V0043JobInfoFlagsItem
from .v0043_job_info_job_state_item import V0043JobInfoJobStateItem
from .v0043_job_info_mail_type_item import V0043JobInfoMailTypeItem
from .v0043_job_info_power import V0043JobInfoPower
from .v0043_job_info_profile_item import V0043JobInfoProfileItem
from .v0043_job_info_shared_item import V0043JobInfoSharedItem
from .v0043_job_mcs import V0043JobMcs
from .v0043_job_required import V0043JobRequired
from .v0043_job_res import V0043JobRes
from .v0043_job_res_core import V0043JobResCore
from .v0043_job_res_core_status_item import V0043JobResCoreStatusItem
from .v0043_job_res_node import V0043JobResNode
from .v0043_job_res_node_cpus import V0043JobResNodeCpus
from .v0043_job_res_node_memory import V0043JobResNodeMemory
from .v0043_job_res_nodes import V0043JobResNodes
from .v0043_job_res_nodes_select_type_item import V0043JobResNodesSelectTypeItem
from .v0043_job_res_select_type_item import V0043JobResSelectTypeItem
from .v0043_job_res_socket import V0043JobResSocket
from .v0043_job_reservation import V0043JobReservation
from .v0043_job_state import V0043JobState
from .v0043_job_state_current_item import V0043JobStateCurrentItem
from .v0043_job_submit_req import V0043JobSubmitReq
from .v0043_job_time import V0043JobTime
from .v0043_job_time_system import V0043JobTimeSystem
from .v0043_job_time_total import V0043JobTimeTotal
from .v0043_job_time_user import V0043JobTimeUser
from .v0043_job_tres import V0043JobTres
from .v0043_kill_jobs_msg import V0043KillJobsMsg
from .v0043_kill_jobs_msg_flags_item import V0043KillJobsMsgFlagsItem
from .v0043_kill_jobs_msg_job_state_item import V0043KillJobsMsgJobStateItem
from .v0043_kill_jobs_resp_job import V0043KillJobsRespJob
from .v0043_kill_jobs_resp_job_error import V0043KillJobsRespJobError
from .v0043_kill_jobs_resp_job_federation import V0043KillJobsRespJobFederation
from .v0043_license import V0043License
from .v0043_node import V0043Node
from .v0043_node_cert_flags_item import V0043NodeCertFlagsItem
from .v0043_node_external_sensors import V0043NodeExternalSensors
from .v0043_node_next_state_after_reboot_item import V0043NodeNextStateAfterRebootItem
from .v0043_node_power import V0043NodePower
from .v0043_node_state_item import V0043NodeStateItem
from .v0043_openapi_accounts_add_cond_resp import V0043OpenapiAccountsAddCondResp
from .v0043_openapi_accounts_add_cond_resp_str import V0043OpenapiAccountsAddCondRespStr
from .v0043_openapi_accounts_removed_resp import V0043OpenapiAccountsRemovedResp
from .v0043_openapi_accounts_resp import V0043OpenapiAccountsResp
from .v0043_openapi_assocs_removed_resp import V0043OpenapiAssocsRemovedResp
from .v0043_openapi_assocs_resp import V0043OpenapiAssocsResp
from .v0043_openapi_clusters_removed_resp import V0043OpenapiClustersRemovedResp
from .v0043_openapi_clusters_resp import V0043OpenapiClustersResp
from .v0043_openapi_diag_resp import V0043OpenapiDiagResp
from .v0043_openapi_error import V0043OpenapiError
from .v0043_openapi_instances_resp import V0043OpenapiInstancesResp
from .v0043_openapi_job_alloc_resp import V0043OpenapiJobAllocResp
from .v0043_openapi_job_info_resp import V0043OpenapiJobInfoResp
from .v0043_openapi_job_post_response import V0043OpenapiJobPostResponse
from .v0043_openapi_job_submit_response import V0043OpenapiJobSubmitResponse
from .v0043_openapi_kill_job_resp import V0043OpenapiKillJobResp
from .v0043_openapi_kill_jobs_resp import V0043OpenapiKillJobsResp
from .v0043_openapi_licenses_resp import V0043OpenapiLicensesResp
from .v0043_openapi_meta import V0043OpenapiMeta
from .v0043_openapi_meta_client import V0043OpenapiMetaClient
from .v0043_openapi_meta_plugin import V0043OpenapiMetaPlugin
from .v0043_openapi_meta_slurm import V0043OpenapiMetaSlurm
from .v0043_openapi_meta_slurm_version import V0043OpenapiMetaSlurmVersion
from .v0043_openapi_nodes_resp import V0043OpenapiNodesResp
from .v0043_openapi_partition_resp import V0043OpenapiPartitionResp
from .v0043_openapi_ping_array_resp import V0043OpenapiPingArrayResp
from .v0043_openapi_reservation_mod_resp import V0043OpenapiReservationModResp
from .v0043_openapi_reservation_resp import V0043OpenapiReservationResp
from .v0043_openapi_resp import V0043OpenapiResp
from .v0043_openapi_shares_resp import V0043OpenapiSharesResp
from .v0043_openapi_slurmdbd_config_resp import V0043OpenapiSlurmdbdConfigResp
from .v0043_openapi_slurmdbd_jobs_resp import V0043OpenapiSlurmdbdJobsResp
from .v0043_openapi_slurmdbd_ping_resp import V0043OpenapiSlurmdbdPingResp
from .v0043_openapi_slurmdbd_qos_removed_resp import V0043OpenapiSlurmdbdQosRemovedResp
from .v0043_openapi_slurmdbd_qos_resp import V0043OpenapiSlurmdbdQosResp
from .v0043_openapi_slurmdbd_stats_resp import V0043OpenapiSlurmdbdStatsResp
from .v0043_openapi_tres_resp import V0043OpenapiTresResp
from .v0043_openapi_users_add_cond_resp import V0043OpenapiUsersAddCondResp
from .v0043_openapi_users_add_cond_resp_str import V0043OpenapiUsersAddCondRespStr
from .v0043_openapi_users_resp import V0043OpenapiUsersResp
from .v0043_openapi_warning import V0043OpenapiWarning
from .v0043_openapi_wckey_removed_resp import V0043OpenapiWckeyRemovedResp
from .v0043_openapi_wckey_resp import V0043OpenapiWckeyResp
from .v0043_part_prio import V0043PartPrio
from .v0043_partition_info import V0043PartitionInfo
from .v0043_partition_info_accounts import V0043PartitionInfoAccounts
from .v0043_partition_info_cpus import V0043PartitionInfoCpus
from .v0043_partition_info_defaults import V0043PartitionInfoDefaults
from .v0043_partition_info_groups import V0043PartitionInfoGroups
from .v0043_partition_info_maximums import V0043PartitionInfoMaximums
from .v0043_partition_info_maximums_oversubscribe import V0043PartitionInfoMaximumsOversubscribe
from .v0043_partition_info_maximums_oversubscribe_flags_item import V0043PartitionInfoMaximumsOversubscribeFlagsItem
from .v0043_partition_info_minimums import V0043PartitionInfoMinimums
from .v0043_partition_info_nodes import V0043PartitionInfoNodes
from .v0043_partition_info_partition import V0043PartitionInfoPartition
from .v0043_partition_info_partition_state_item import V0043PartitionInfoPartitionStateItem
from .v0043_partition_info_priority import V0043PartitionInfoPriority
from .v0043_partition_info_qos import V0043PartitionInfoQos
from .v0043_partition_info_select_type_item import V0043PartitionInfoSelectTypeItem
from .v0043_partition_info_timeouts import V0043PartitionInfoTimeouts
from .v0043_partition_info_tres import V0043PartitionInfoTres
from .v0043_process_exit_code_verbose import V0043ProcessExitCodeVerbose
from .v0043_process_exit_code_verbose_signal import V0043ProcessExitCodeVerboseSignal
from .v0043_process_exit_code_verbose_status_item import V0043ProcessExitCodeVerboseStatusItem
from .v0043_qos import V0043Qos
from .v0043_qos_flags_item import V0043QosFlagsItem
from .v0043_qos_limits import V0043QosLimits
from .v0043_qos_limits_max import V0043QosLimitsMax
from .v0043_qos_limits_max_accruing import V0043QosLimitsMaxAccruing
from .v0043_qos_limits_max_accruing_per import V0043QosLimitsMaxAccruingPer
from .v0043_qos_limits_max_active_jobs import V0043QosLimitsMaxActiveJobs
from .v0043_qos_limits_max_jobs import V0043QosLimitsMaxJobs
from .v0043_qos_limits_max_jobs_active_jobs import V0043QosLimitsMaxJobsActiveJobs
from .v0043_qos_limits_max_jobs_active_jobs_per import V0043QosLimitsMaxJobsActiveJobsPer
from .v0043_qos_limits_max_jobs_per import V0043QosLimitsMaxJobsPer
from .v0043_qos_limits_max_tres import V0043QosLimitsMaxTres
from .v0043_qos_limits_max_tres_minutes import V0043QosLimitsMaxTresMinutes
from .v0043_qos_limits_max_tres_minutes_per import V0043QosLimitsMaxTresMinutesPer
from .v0043_qos_limits_max_tres_per import V0043QosLimitsMaxTresPer
from .v0043_qos_limits_max_wall_clock import V0043QosLimitsMaxWallClock
from .v0043_qos_limits_max_wall_clock_per import V0043QosLimitsMaxWallClockPer
from .v0043_qos_limits_min import V0043QosLimitsMin
from .v0043_qos_limits_min_tres import V0043QosLimitsMinTres
from .v0043_qos_limits_min_tres_per import V0043QosLimitsMinTresPer
from .v0043_qos_preempt import V0043QosPreempt
from .v0043_qos_preempt_mode_item import V0043QosPreemptModeItem
from .v0043_reservation_core_spec import V0043ReservationCoreSpec
from .v0043_reservation_desc_msg import V0043ReservationDescMsg
from .v0043_reservation_desc_msg_flags_item import V0043ReservationDescMsgFlagsItem
from .v0043_reservation_desc_msg_purge_completed import V0043ReservationDescMsgPurgeCompleted
from .v0043_reservation_info import V0043ReservationInfo
from .v0043_reservation_info_flags_item import V0043ReservationInfoFlagsItem
from .v0043_reservation_info_purge_completed import V0043ReservationInfoPurgeCompleted
from .v0043_reservation_mod_req import V0043ReservationModReq
from .v0043_rollup_stats import V0043RollupStats
from .v0043_rollup_stats_daily import V0043RollupStatsDaily
from .v0043_rollup_stats_daily_duration import V0043RollupStatsDailyDuration
from .v0043_rollup_stats_hourly import V0043RollupStatsHourly
from .v0043_rollup_stats_hourly_duration import V0043RollupStatsHourlyDuration
from .v0043_rollup_stats_monthly import V0043RollupStatsMonthly
from .v0043_rollup_stats_monthly_duration import V0043RollupStatsMonthlyDuration
from .v0043_schedule_exit_fields import V0043ScheduleExitFields
from .v0043_shares_float_128_tres import V0043SharesFloat128Tres
from .v0043_shares_resp_msg import V0043SharesRespMsg
from .v0043_shares_uint_64_tres import V0043SharesUint64Tres
from .v0043_slurmdbd_ping import V0043SlurmdbdPing
from .v0043_stats_msg import V0043StatsMsg
from .v0043_stats_msg_rpc_dump import V0043StatsMsgRpcDump
from .v0043_stats_msg_rpc_queue import V0043StatsMsgRpcQueue
from .v0043_stats_msg_rpc_type import V0043StatsMsgRpcType
from .v0043_stats_msg_rpc_user import V0043StatsMsgRpcUser
from .v0043_stats_rec import V0043StatsRec
from .v0043_stats_rpc import V0043StatsRpc
from .v0043_stats_rpc_time import V0043StatsRpcTime
from .v0043_stats_user import V0043StatsUser
from .v0043_stats_user_time import V0043StatsUserTime
from .v0043_step import V0043Step
from .v0043_step_cpu import V0043StepCPU
from .v0043_step_cpu_requested_frequency import V0043StepCPURequestedFrequency
from .v0043_step_nodes import V0043StepNodes
from .v0043_step_state_item import V0043StepStateItem
from .v0043_step_statistics import V0043StepStatistics
from .v0043_step_statistics_cpu import V0043StepStatisticsCPU
from .v0043_step_statistics_energy import V0043StepStatisticsEnergy
from .v0043_step_step import V0043StepStep
from .v0043_step_task import V0043StepTask
from .v0043_step_tasks import V0043StepTasks
from .v0043_step_time import V0043StepTime
from .v0043_step_time_system import V0043StepTimeSystem
from .v0043_step_time_total import V0043StepTimeTotal
from .v0043_step_time_user import V0043StepTimeUser
from .v0043_step_tres import V0043StepTres
from .v0043_step_tres_consumed import V0043StepTresConsumed
from .v0043_step_tres_requested import V0043StepTresRequested
from .v0043_tres import V0043Tres
from .v0043_uint_16_no_val_struct import V0043Uint16NoValStruct
from .v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
from .v0043_uint_64_no_val_struct import V0043Uint64NoValStruct
from .v0043_update_node_msg import V0043UpdateNodeMsg
from .v0043_update_node_msg_state_item import V0043UpdateNodeMsgStateItem
from .v0043_user import V0043User
from .v0043_user_administrator_level_item import V0043UserAdministratorLevelItem
from .v0043_user_default import V0043UserDefault
from .v0043_user_flags_item import V0043UserFlagsItem
from .v0043_user_short import V0043UserShort
from .v0043_user_short_adminlevel_item import V0043UserShortAdminlevelItem
from .v0043_users_add_cond import V0043UsersAddCond
from .v0043_wckey import V0043Wckey
from .v0043_wckey_flags_item import V0043WckeyFlagsItem
from .v0043_wckey_tag_struct import V0043WckeyTagStruct
from .v0043_wckey_tag_struct_flags_item import V0043WckeyTagStructFlagsItem

__all__ = (
    "SlurmdbV0043DeleteClusterClassification",
    "SlurmdbV0043DeleteClusterFlags",
    "SlurmdbV0043GetClusterClassification",
    "SlurmdbV0043GetClusterFlags",
    "SlurmdbV0043GetQosPreemptMode",
    "SlurmdbV0043GetUsersAdminLevel",
    "SlurmdbV0043PostQosPreemptMode",
    "SlurmdbV0043PostUsersAssociationFlags",
    "SlurmV0043DeleteJobFlags",
    "SlurmV0043GetJobFlags",
    "SlurmV0043GetJobsFlags",
    "SlurmV0043GetNodeFlags",
    "SlurmV0043GetNodesFlags",
    "SlurmV0043GetPartitionFlags",
    "SlurmV0043GetPartitionsFlags",
    "V0043Account",
    "V0043AccountFlagsItem",
    "V0043Accounting",
    "V0043AccountingAllocated",
    "V0043AccountsAddCond",
    "V0043AccountShort",
    "V0043AcctGatherEnergy",
    "V0043Assoc",
    "V0043AssocDefault",
    "V0043AssocFlagsItem",
    "V0043AssocMax",
    "V0043AssocMaxJobs",
    "V0043AssocMaxJobsPer",
    "V0043AssocMaxPer",
    "V0043AssocMaxPerAccount",
    "V0043AssocMaxTres",
    "V0043AssocMaxTresGroup",
    "V0043AssocMaxTresMinutes",
    "V0043AssocMaxTresMinutesPer",
    "V0043AssocMaxTresPer",
    "V0043AssocMin",
    "V0043AssocRecSet",
    "V0043AssocSharesObjWrap",
    "V0043AssocSharesObjWrapFairshare",
    "V0043AssocSharesObjWrapTres",
    "V0043AssocSharesObjWrapTypeItem",
    "V0043AssocShort",
    "V0043BfExitFields",
    "V0043ClusterRec",
    "V0043ClusterRecAssociations",
    "V0043ClusterRecController",
    "V0043ClusterRecFlagsItem",
    "V0043ControllerPing",
    "V0043Coord",
    "V0043CronEntry",
    "V0043CronEntryFlagsItem",
    "V0043CronEntryLine",
    "V0043Float64NoValStruct",
    "V0043Instance",
    "V0043InstanceTime",
    "V0043Job",
    "V0043JobAllocReq",
    "V0043JobArray",
    "V0043JobArrayLimits",
    "V0043JobArrayLimitsMax",
    "V0043JobArrayLimitsMaxRunning",
    "V0043JobArrayResponseMsgEntry",
    "V0043JobComment",
    "V0043JobDescMsg",
    "V0043JobDescMsgCpuBindingFlagsItem",
    "V0043JobDescMsgFlagsItem",
    "V0043JobDescMsgKillWarningFlagsItem",
    "V0043JobDescMsgMailTypeItem",
    "V0043JobDescMsgMemoryBindingTypeItem",
    "V0043JobDescMsgOpenModeItem",
    "V0043JobDescMsgProfileItem",
    "V0043JobDescMsgRlimits",
    "V0043JobDescMsgSharedItem",
    "V0043JobDescMsgX11Item",
    "V0043JobFlagsItem",
    "V0043JobHet",
    "V0043JobInfo",
    "V0043JobInfoFlagsItem",
    "V0043JobInfoJobStateItem",
    "V0043JobInfoMailTypeItem",
    "V0043JobInfoPower",
    "V0043JobInfoProfileItem",
    "V0043JobInfoSharedItem",
    "V0043JobMcs",
    "V0043JobRequired",
    "V0043JobRes",
    "V0043JobResCore",
    "V0043JobResCoreStatusItem",
    "V0043JobReservation",
    "V0043JobResNode",
    "V0043JobResNodeCpus",
    "V0043JobResNodeMemory",
    "V0043JobResNodes",
    "V0043JobResNodesSelectTypeItem",
    "V0043JobResSelectTypeItem",
    "V0043JobResSocket",
    "V0043JobState",
    "V0043JobStateCurrentItem",
    "V0043JobSubmitReq",
    "V0043JobTime",
    "V0043JobTimeSystem",
    "V0043JobTimeTotal",
    "V0043JobTimeUser",
    "V0043JobTres",
    "V0043KillJobsMsg",
    "V0043KillJobsMsgFlagsItem",
    "V0043KillJobsMsgJobStateItem",
    "V0043KillJobsRespJob",
    "V0043KillJobsRespJobError",
    "V0043KillJobsRespJobFederation",
    "V0043License",
    "V0043Node",
    "V0043NodeCertFlagsItem",
    "V0043NodeExternalSensors",
    "V0043NodeNextStateAfterRebootItem",
    "V0043NodePower",
    "V0043NodeStateItem",
    "V0043OpenapiAccountsAddCondResp",
    "V0043OpenapiAccountsAddCondRespStr",
    "V0043OpenapiAccountsRemovedResp",
    "V0043OpenapiAccountsResp",
    "V0043OpenapiAssocsRemovedResp",
    "V0043OpenapiAssocsResp",
    "V0043OpenapiClustersRemovedResp",
    "V0043OpenapiClustersResp",
    "V0043OpenapiDiagResp",
    "V0043OpenapiError",
    "V0043OpenapiInstancesResp",
    "V0043OpenapiJobAllocResp",
    "V0043OpenapiJobInfoResp",
    "V0043OpenapiJobPostResponse",
    "V0043OpenapiJobSubmitResponse",
    "V0043OpenapiKillJobResp",
    "V0043OpenapiKillJobsResp",
    "V0043OpenapiLicensesResp",
    "V0043OpenapiMeta",
    "V0043OpenapiMetaClient",
    "V0043OpenapiMetaPlugin",
    "V0043OpenapiMetaSlurm",
    "V0043OpenapiMetaSlurmVersion",
    "V0043OpenapiNodesResp",
    "V0043OpenapiPartitionResp",
    "V0043OpenapiPingArrayResp",
    "V0043OpenapiReservationModResp",
    "V0043OpenapiReservationResp",
    "V0043OpenapiResp",
    "V0043OpenapiSharesResp",
    "V0043OpenapiSlurmdbdConfigResp",
    "V0043OpenapiSlurmdbdJobsResp",
    "V0043OpenapiSlurmdbdPingResp",
    "V0043OpenapiSlurmdbdQosRemovedResp",
    "V0043OpenapiSlurmdbdQosResp",
    "V0043OpenapiSlurmdbdStatsResp",
    "V0043OpenapiTresResp",
    "V0043OpenapiUsersAddCondResp",
    "V0043OpenapiUsersAddCondRespStr",
    "V0043OpenapiUsersResp",
    "V0043OpenapiWarning",
    "V0043OpenapiWckeyRemovedResp",
    "V0043OpenapiWckeyResp",
    "V0043PartitionInfo",
    "V0043PartitionInfoAccounts",
    "V0043PartitionInfoCpus",
    "V0043PartitionInfoDefaults",
    "V0043PartitionInfoGroups",
    "V0043PartitionInfoMaximums",
    "V0043PartitionInfoMaximumsOversubscribe",
    "V0043PartitionInfoMaximumsOversubscribeFlagsItem",
    "V0043PartitionInfoMinimums",
    "V0043PartitionInfoNodes",
    "V0043PartitionInfoPartition",
    "V0043PartitionInfoPartitionStateItem",
    "V0043PartitionInfoPriority",
    "V0043PartitionInfoQos",
    "V0043PartitionInfoSelectTypeItem",
    "V0043PartitionInfoTimeouts",
    "V0043PartitionInfoTres",
    "V0043PartPrio",
    "V0043ProcessExitCodeVerbose",
    "V0043ProcessExitCodeVerboseSignal",
    "V0043ProcessExitCodeVerboseStatusItem",
    "V0043Qos",
    "V0043QosFlagsItem",
    "V0043QosLimits",
    "V0043QosLimitsMax",
    "V0043QosLimitsMaxAccruing",
    "V0043QosLimitsMaxAccruingPer",
    "V0043QosLimitsMaxActiveJobs",
    "V0043QosLimitsMaxJobs",
    "V0043QosLimitsMaxJobsActiveJobs",
    "V0043QosLimitsMaxJobsActiveJobsPer",
    "V0043QosLimitsMaxJobsPer",
    "V0043QosLimitsMaxTres",
    "V0043QosLimitsMaxTresMinutes",
    "V0043QosLimitsMaxTresMinutesPer",
    "V0043QosLimitsMaxTresPer",
    "V0043QosLimitsMaxWallClock",
    "V0043QosLimitsMaxWallClockPer",
    "V0043QosLimitsMin",
    "V0043QosLimitsMinTres",
    "V0043QosLimitsMinTresPer",
    "V0043QosPreempt",
    "V0043QosPreemptModeItem",
    "V0043ReservationCoreSpec",
    "V0043ReservationDescMsg",
    "V0043ReservationDescMsgFlagsItem",
    "V0043ReservationDescMsgPurgeCompleted",
    "V0043ReservationInfo",
    "V0043ReservationInfoFlagsItem",
    "V0043ReservationInfoPurgeCompleted",
    "V0043ReservationModReq",
    "V0043RollupStats",
    "V0043RollupStatsDaily",
    "V0043RollupStatsDailyDuration",
    "V0043RollupStatsHourly",
    "V0043RollupStatsHourlyDuration",
    "V0043RollupStatsMonthly",
    "V0043RollupStatsMonthlyDuration",
    "V0043ScheduleExitFields",
    "V0043SharesFloat128Tres",
    "V0043SharesRespMsg",
    "V0043SharesUint64Tres",
    "V0043SlurmdbdPing",
    "V0043StatsMsg",
    "V0043StatsMsgRpcDump",
    "V0043StatsMsgRpcQueue",
    "V0043StatsMsgRpcType",
    "V0043StatsMsgRpcUser",
    "V0043StatsRec",
    "V0043StatsRpc",
    "V0043StatsRpcTime",
    "V0043StatsUser",
    "V0043StatsUserTime",
    "V0043Step",
    "V0043StepCPU",
    "V0043StepCPURequestedFrequency",
    "V0043StepNodes",
    "V0043StepStateItem",
    "V0043StepStatistics",
    "V0043StepStatisticsCPU",
    "V0043StepStatisticsEnergy",
    "V0043StepStep",
    "V0043StepTask",
    "V0043StepTasks",
    "V0043StepTime",
    "V0043StepTimeSystem",
    "V0043StepTimeTotal",
    "V0043StepTimeUser",
    "V0043StepTres",
    "V0043StepTresConsumed",
    "V0043StepTresRequested",
    "V0043Tres",
    "V0043Uint16NoValStruct",
    "V0043Uint32NoValStruct",
    "V0043Uint64NoValStruct",
    "V0043UpdateNodeMsg",
    "V0043UpdateNodeMsgStateItem",
    "V0043User",
    "V0043UserAdministratorLevelItem",
    "V0043UserDefault",
    "V0043UserFlagsItem",
    "V0043UsersAddCond",
    "V0043UserShort",
    "V0043UserShortAdminlevelItem",
    "V0043Wckey",
    "V0043WckeyFlagsItem",
    "V0043WckeyTagStruct",
    "V0043WckeyTagStructFlagsItem",
)
