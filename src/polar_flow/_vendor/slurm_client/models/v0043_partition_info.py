from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_partition_info_select_type_item import V0043PartitionInfoSelectTypeItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_partition_info_accounts import V0043PartitionInfoAccounts
    from ..models.v0043_partition_info_cpus import V0043PartitionInfoCpus
    from ..models.v0043_partition_info_defaults import V0043PartitionInfoDefaults
    from ..models.v0043_partition_info_groups import V0043PartitionInfoGroups
    from ..models.v0043_partition_info_maximums import V0043PartitionInfoMaximums
    from ..models.v0043_partition_info_minimums import V0043PartitionInfoMinimums
    from ..models.v0043_partition_info_nodes import V0043PartitionInfoNodes
    from ..models.v0043_partition_info_partition import V0043PartitionInfoPartition
    from ..models.v0043_partition_info_priority import V0043PartitionInfoPriority
    from ..models.v0043_partition_info_qos import V0043PartitionInfoQos
    from ..models.v0043_partition_info_timeouts import V0043PartitionInfoTimeouts
    from ..models.v0043_partition_info_tres import V0043PartitionInfoTres
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043PartitionInfo")


@_attrs_define
class V0043PartitionInfo:
    """
    Attributes:
        nodes (Union[Unset, V0043PartitionInfoNodes]):
        accounts (Union[Unset, V0043PartitionInfoAccounts]):
        groups (Union[Unset, V0043PartitionInfoGroups]):
        qos (Union[Unset, V0043PartitionInfoQos]):
        alternate (Union[Unset, str]): Alternate - Partition name of alternate partition to be used if the state of this
            partition is DRAIN or INACTIVE
        tres (Union[Unset, V0043PartitionInfoTres]):
        cluster (Union[Unset, str]): Cluster name
        select_type (Union[Unset, list[V0043PartitionInfoSelectTypeItem]]): Scheduler consumable resource selection type
        cpus (Union[Unset, V0043PartitionInfoCpus]):
        defaults (Union[Unset, V0043PartitionInfoDefaults]):
        grace_time (Union[Unset, int]): GraceTime - Grace time in seconds to be extended to a job which has been
            selected for preemption
        maximums (Union[Unset, V0043PartitionInfoMaximums]):
        minimums (Union[Unset, V0043PartitionInfoMinimums]):
        name (Union[Unset, str]): PartitionName - Name by which the partition may be referenced
        node_sets (Union[Unset, str]): NodeSets - Comma-separated list of nodesets which are associated with this
            partition
        priority (Union[Unset, V0043PartitionInfoPriority]):
        timeouts (Union[Unset, V0043PartitionInfoTimeouts]):
        topology (Union[Unset, str]): Topology - Name of the topology, defined in topology.yaml, used by jobs in this
            partition
        partition (Union[Unset, V0043PartitionInfoPartition]):
        suspend_time (Union[Unset, V0043Uint32NoValStruct]):
    """

    nodes: Union[Unset, "V0043PartitionInfoNodes"] = UNSET
    accounts: Union[Unset, "V0043PartitionInfoAccounts"] = UNSET
    groups: Union[Unset, "V0043PartitionInfoGroups"] = UNSET
    qos: Union[Unset, "V0043PartitionInfoQos"] = UNSET
    alternate: Union[Unset, str] = UNSET
    tres: Union[Unset, "V0043PartitionInfoTres"] = UNSET
    cluster: Union[Unset, str] = UNSET
    select_type: Union[Unset, list[V0043PartitionInfoSelectTypeItem]] = UNSET
    cpus: Union[Unset, "V0043PartitionInfoCpus"] = UNSET
    defaults: Union[Unset, "V0043PartitionInfoDefaults"] = UNSET
    grace_time: Union[Unset, int] = UNSET
    maximums: Union[Unset, "V0043PartitionInfoMaximums"] = UNSET
    minimums: Union[Unset, "V0043PartitionInfoMinimums"] = UNSET
    name: Union[Unset, str] = UNSET
    node_sets: Union[Unset, str] = UNSET
    priority: Union[Unset, "V0043PartitionInfoPriority"] = UNSET
    timeouts: Union[Unset, "V0043PartitionInfoTimeouts"] = UNSET
    topology: Union[Unset, str] = UNSET
    partition: Union[Unset, "V0043PartitionInfoPartition"] = UNSET
    suspend_time: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        nodes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.nodes, Unset):
            nodes = self.nodes.to_dict()

        accounts: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.accounts, Unset):
            accounts = self.accounts.to_dict()

        groups: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.groups, Unset):
            groups = self.groups.to_dict()

        qos: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.qos, Unset):
            qos = self.qos.to_dict()

        alternate = self.alternate

        tres: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = self.tres.to_dict()

        cluster = self.cluster

        select_type: Union[Unset, list[str]] = UNSET
        if not isinstance(self.select_type, Unset):
            select_type = []
            for select_type_item_data in self.select_type:
                select_type_item = select_type_item_data.value
                select_type.append(select_type_item)

        cpus: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.cpus, Unset):
            cpus = self.cpus.to_dict()

        defaults: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.defaults, Unset):
            defaults = self.defaults.to_dict()

        grace_time = self.grace_time

        maximums: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.maximums, Unset):
            maximums = self.maximums.to_dict()

        minimums: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.minimums, Unset):
            minimums = self.minimums.to_dict()

        name = self.name

        node_sets = self.node_sets

        priority: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.priority, Unset):
            priority = self.priority.to_dict()

        timeouts: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.timeouts, Unset):
            timeouts = self.timeouts.to_dict()

        topology = self.topology

        partition: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.partition, Unset):
            partition = self.partition.to_dict()

        suspend_time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.suspend_time, Unset):
            suspend_time = self.suspend_time.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if nodes is not UNSET:
            field_dict["nodes"] = nodes
        if accounts is not UNSET:
            field_dict["accounts"] = accounts
        if groups is not UNSET:
            field_dict["groups"] = groups
        if qos is not UNSET:
            field_dict["qos"] = qos
        if alternate is not UNSET:
            field_dict["alternate"] = alternate
        if tres is not UNSET:
            field_dict["tres"] = tres
        if cluster is not UNSET:
            field_dict["cluster"] = cluster
        if select_type is not UNSET:
            field_dict["select_type"] = select_type
        if cpus is not UNSET:
            field_dict["cpus"] = cpus
        if defaults is not UNSET:
            field_dict["defaults"] = defaults
        if grace_time is not UNSET:
            field_dict["grace_time"] = grace_time
        if maximums is not UNSET:
            field_dict["maximums"] = maximums
        if minimums is not UNSET:
            field_dict["minimums"] = minimums
        if name is not UNSET:
            field_dict["name"] = name
        if node_sets is not UNSET:
            field_dict["node_sets"] = node_sets
        if priority is not UNSET:
            field_dict["priority"] = priority
        if timeouts is not UNSET:
            field_dict["timeouts"] = timeouts
        if topology is not UNSET:
            field_dict["topology"] = topology
        if partition is not UNSET:
            field_dict["partition"] = partition
        if suspend_time is not UNSET:
            field_dict["suspend_time"] = suspend_time

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_partition_info_accounts import V0043PartitionInfoAccounts
        from ..models.v0043_partition_info_cpus import V0043PartitionInfoCpus
        from ..models.v0043_partition_info_defaults import V0043PartitionInfoDefaults
        from ..models.v0043_partition_info_groups import V0043PartitionInfoGroups
        from ..models.v0043_partition_info_maximums import V0043PartitionInfoMaximums
        from ..models.v0043_partition_info_minimums import V0043PartitionInfoMinimums
        from ..models.v0043_partition_info_nodes import V0043PartitionInfoNodes
        from ..models.v0043_partition_info_partition import V0043PartitionInfoPartition
        from ..models.v0043_partition_info_priority import V0043PartitionInfoPriority
        from ..models.v0043_partition_info_qos import V0043PartitionInfoQos
        from ..models.v0043_partition_info_timeouts import V0043PartitionInfoTimeouts
        from ..models.v0043_partition_info_tres import V0043PartitionInfoTres
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        _nodes = d.pop("nodes", UNSET)
        nodes: Union[Unset, V0043PartitionInfoNodes]
        if isinstance(_nodes, Unset):
            nodes = UNSET
        else:
            nodes = V0043PartitionInfoNodes.from_dict(_nodes)

        _accounts = d.pop("accounts", UNSET)
        accounts: Union[Unset, V0043PartitionInfoAccounts]
        if isinstance(_accounts, Unset):
            accounts = UNSET
        else:
            accounts = V0043PartitionInfoAccounts.from_dict(_accounts)

        _groups = d.pop("groups", UNSET)
        groups: Union[Unset, V0043PartitionInfoGroups]
        if isinstance(_groups, Unset):
            groups = UNSET
        else:
            groups = V0043PartitionInfoGroups.from_dict(_groups)

        _qos = d.pop("qos", UNSET)
        qos: Union[Unset, V0043PartitionInfoQos]
        if isinstance(_qos, Unset):
            qos = UNSET
        else:
            qos = V0043PartitionInfoQos.from_dict(_qos)

        alternate = d.pop("alternate", UNSET)

        _tres = d.pop("tres", UNSET)
        tres: Union[Unset, V0043PartitionInfoTres]
        if isinstance(_tres, Unset):
            tres = UNSET
        else:
            tres = V0043PartitionInfoTres.from_dict(_tres)

        cluster = d.pop("cluster", UNSET)

        select_type = []
        _select_type = d.pop("select_type", UNSET)
        for select_type_item_data in _select_type or []:
            select_type_item = V0043PartitionInfoSelectTypeItem(select_type_item_data)

            select_type.append(select_type_item)

        _cpus = d.pop("cpus", UNSET)
        cpus: Union[Unset, V0043PartitionInfoCpus]
        if isinstance(_cpus, Unset):
            cpus = UNSET
        else:
            cpus = V0043PartitionInfoCpus.from_dict(_cpus)

        _defaults = d.pop("defaults", UNSET)
        defaults: Union[Unset, V0043PartitionInfoDefaults]
        if isinstance(_defaults, Unset):
            defaults = UNSET
        else:
            defaults = V0043PartitionInfoDefaults.from_dict(_defaults)

        grace_time = d.pop("grace_time", UNSET)

        _maximums = d.pop("maximums", UNSET)
        maximums: Union[Unset, V0043PartitionInfoMaximums]
        if isinstance(_maximums, Unset):
            maximums = UNSET
        else:
            maximums = V0043PartitionInfoMaximums.from_dict(_maximums)

        _minimums = d.pop("minimums", UNSET)
        minimums: Union[Unset, V0043PartitionInfoMinimums]
        if isinstance(_minimums, Unset):
            minimums = UNSET
        else:
            minimums = V0043PartitionInfoMinimums.from_dict(_minimums)

        name = d.pop("name", UNSET)

        node_sets = d.pop("node_sets", UNSET)

        _priority = d.pop("priority", UNSET)
        priority: Union[Unset, V0043PartitionInfoPriority]
        if isinstance(_priority, Unset):
            priority = UNSET
        else:
            priority = V0043PartitionInfoPriority.from_dict(_priority)

        _timeouts = d.pop("timeouts", UNSET)
        timeouts: Union[Unset, V0043PartitionInfoTimeouts]
        if isinstance(_timeouts, Unset):
            timeouts = UNSET
        else:
            timeouts = V0043PartitionInfoTimeouts.from_dict(_timeouts)

        topology = d.pop("topology", UNSET)

        _partition = d.pop("partition", UNSET)
        partition: Union[Unset, V0043PartitionInfoPartition]
        if isinstance(_partition, Unset):
            partition = UNSET
        else:
            partition = V0043PartitionInfoPartition.from_dict(_partition)

        _suspend_time = d.pop("suspend_time", UNSET)
        suspend_time: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_suspend_time, Unset):
            suspend_time = UNSET
        else:
            suspend_time = V0043Uint32NoValStruct.from_dict(_suspend_time)

        v0043_partition_info = cls(
            nodes=nodes,
            accounts=accounts,
            groups=groups,
            qos=qos,
            alternate=alternate,
            tres=tres,
            cluster=cluster,
            select_type=select_type,
            cpus=cpus,
            defaults=defaults,
            grace_time=grace_time,
            maximums=maximums,
            minimums=minimums,
            name=name,
            node_sets=node_sets,
            priority=priority,
            timeouts=timeouts,
            topology=topology,
            partition=partition,
            suspend_time=suspend_time,
        )

        v0043_partition_info.additional_properties = d
        return v0043_partition_info

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
