from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_reservation_desc_msg_flags_item import V0043ReservationDescMsgFlagsItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_reservation_desc_msg_purge_completed import V0043ReservationDescMsgPurgeCompleted
    from ..models.v0043_tres import V0043Tres
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043ReservationDescMsg")


@_attrs_define
class V0043ReservationDescMsg:
    """
    Attributes:
        accounts (Union[Unset, list[str]]):
        burst_buffer (Union[Unset, str]): BurstBuffer
        comment (Union[Unset, str]): Arbitrary string
        core_count (Union[Unset, V0043Uint32NoValStruct]):
        duration (Union[Unset, V0043Uint32NoValStruct]):
        end_time (Union[Unset, V0043Uint64NoValStruct]):
        features (Union[Unset, str]): Requested node features. Multiple values may be "&" separated if all features are
            required (AND operation) or separated by "|" if any of the specified features are required (OR operation).
            Parenthesis are also supported for features to be ANDed together with counts of nodes having the specified
            features.
        flags (Union[Unset, list[V0043ReservationDescMsgFlagsItem]]): Flags associated with this reservation. Note, to
            remove flags use "NO_" prefixed flag excluding NO_HOLD_JOBS_AFTER_END
        groups (Union[Unset, list[str]]):
        licenses (Union[Unset, list[str]]):
        max_start_delay (Union[Unset, V0043Uint32NoValStruct]):
        name (Union[Unset, str]): ReservationName
        node_count (Union[Unset, V0043Uint32NoValStruct]):
        node_list (Union[Unset, list[str]]):
        partition (Union[Unset, str]): Partition used to reserve nodes from. This will attempt to allocate all nodes in
            the specified partition unless you request fewer resources than are available with core_cnt, node_cnt or tres.
        purge_completed (Union[Unset, V0043ReservationDescMsgPurgeCompleted]):
        start_time (Union[Unset, V0043Uint64NoValStruct]):
        tres (Union[Unset, list['V0043Tres']]):
        users (Union[Unset, list[str]]):
    """

    accounts: Union[Unset, list[str]] = UNSET
    burst_buffer: Union[Unset, str] = UNSET
    comment: Union[Unset, str] = UNSET
    core_count: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    duration: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    end_time: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    features: Union[Unset, str] = UNSET
    flags: Union[Unset, list[V0043ReservationDescMsgFlagsItem]] = UNSET
    groups: Union[Unset, list[str]] = UNSET
    licenses: Union[Unset, list[str]] = UNSET
    max_start_delay: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    name: Union[Unset, str] = UNSET
    node_count: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    node_list: Union[Unset, list[str]] = UNSET
    partition: Union[Unset, str] = UNSET
    purge_completed: Union[Unset, "V0043ReservationDescMsgPurgeCompleted"] = UNSET
    start_time: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    tres: Union[Unset, list["V0043Tres"]] = UNSET
    users: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        accounts: Union[Unset, list[str]] = UNSET
        if not isinstance(self.accounts, Unset):
            accounts = self.accounts

        burst_buffer = self.burst_buffer

        comment = self.comment

        core_count: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.core_count, Unset):
            core_count = self.core_count.to_dict()

        duration: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.duration, Unset):
            duration = self.duration.to_dict()

        end_time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.end_time, Unset):
            end_time = self.end_time.to_dict()

        features = self.features

        flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.flags, Unset):
            flags = []
            for flags_item_data in self.flags:
                flags_item = flags_item_data.value
                flags.append(flags_item)

        groups: Union[Unset, list[str]] = UNSET
        if not isinstance(self.groups, Unset):
            groups = self.groups

        licenses: Union[Unset, list[str]] = UNSET
        if not isinstance(self.licenses, Unset):
            licenses = self.licenses

        max_start_delay: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.max_start_delay, Unset):
            max_start_delay = self.max_start_delay.to_dict()

        name = self.name

        node_count: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.node_count, Unset):
            node_count = self.node_count.to_dict()

        node_list: Union[Unset, list[str]] = UNSET
        if not isinstance(self.node_list, Unset):
            node_list = self.node_list

        partition = self.partition

        purge_completed: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.purge_completed, Unset):
            purge_completed = self.purge_completed.to_dict()

        start_time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.start_time, Unset):
            start_time = self.start_time.to_dict()

        tres: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = []
            for componentsschemasv0_0_43_tres_list_item_data in self.tres:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                tres.append(componentsschemasv0_0_43_tres_list_item)

        users: Union[Unset, list[str]] = UNSET
        if not isinstance(self.users, Unset):
            users = self.users

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if accounts is not UNSET:
            field_dict["accounts"] = accounts
        if burst_buffer is not UNSET:
            field_dict["burst_buffer"] = burst_buffer
        if comment is not UNSET:
            field_dict["comment"] = comment
        if core_count is not UNSET:
            field_dict["core_count"] = core_count
        if duration is not UNSET:
            field_dict["duration"] = duration
        if end_time is not UNSET:
            field_dict["end_time"] = end_time
        if features is not UNSET:
            field_dict["features"] = features
        if flags is not UNSET:
            field_dict["flags"] = flags
        if groups is not UNSET:
            field_dict["groups"] = groups
        if licenses is not UNSET:
            field_dict["licenses"] = licenses
        if max_start_delay is not UNSET:
            field_dict["max_start_delay"] = max_start_delay
        if name is not UNSET:
            field_dict["name"] = name
        if node_count is not UNSET:
            field_dict["node_count"] = node_count
        if node_list is not UNSET:
            field_dict["node_list"] = node_list
        if partition is not UNSET:
            field_dict["partition"] = partition
        if purge_completed is not UNSET:
            field_dict["purge_completed"] = purge_completed
        if start_time is not UNSET:
            field_dict["start_time"] = start_time
        if tres is not UNSET:
            field_dict["tres"] = tres
        if users is not UNSET:
            field_dict["users"] = users

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_reservation_desc_msg_purge_completed import V0043ReservationDescMsgPurgeCompleted
        from ..models.v0043_tres import V0043Tres
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        accounts = cast(list[str], d.pop("accounts", UNSET))

        burst_buffer = d.pop("burst_buffer", UNSET)

        comment = d.pop("comment", UNSET)

        _core_count = d.pop("core_count", UNSET)
        core_count: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_core_count, Unset):
            core_count = UNSET
        else:
            core_count = V0043Uint32NoValStruct.from_dict(_core_count)

        _duration = d.pop("duration", UNSET)
        duration: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_duration, Unset):
            duration = UNSET
        else:
            duration = V0043Uint32NoValStruct.from_dict(_duration)

        _end_time = d.pop("end_time", UNSET)
        end_time: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_end_time, Unset):
            end_time = UNSET
        else:
            end_time = V0043Uint64NoValStruct.from_dict(_end_time)

        features = d.pop("features", UNSET)

        flags = []
        _flags = d.pop("flags", UNSET)
        for flags_item_data in _flags or []:
            flags_item = V0043ReservationDescMsgFlagsItem(flags_item_data)

            flags.append(flags_item)

        groups = cast(list[str], d.pop("groups", UNSET))

        licenses = cast(list[str], d.pop("licenses", UNSET))

        _max_start_delay = d.pop("max_start_delay", UNSET)
        max_start_delay: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_max_start_delay, Unset):
            max_start_delay = UNSET
        else:
            max_start_delay = V0043Uint32NoValStruct.from_dict(_max_start_delay)

        name = d.pop("name", UNSET)

        _node_count = d.pop("node_count", UNSET)
        node_count: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_node_count, Unset):
            node_count = UNSET
        else:
            node_count = V0043Uint32NoValStruct.from_dict(_node_count)

        node_list = cast(list[str], d.pop("node_list", UNSET))

        partition = d.pop("partition", UNSET)

        _purge_completed = d.pop("purge_completed", UNSET)
        purge_completed: Union[Unset, V0043ReservationDescMsgPurgeCompleted]
        if isinstance(_purge_completed, Unset):
            purge_completed = UNSET
        else:
            purge_completed = V0043ReservationDescMsgPurgeCompleted.from_dict(_purge_completed)

        _start_time = d.pop("start_time", UNSET)
        start_time: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_start_time, Unset):
            start_time = UNSET
        else:
            start_time = V0043Uint64NoValStruct.from_dict(_start_time)

        tres = []
        _tres = d.pop("tres", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _tres or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            tres.append(componentsschemasv0_0_43_tres_list_item)

        users = cast(list[str], d.pop("users", UNSET))

        v0043_reservation_desc_msg = cls(
            accounts=accounts,
            burst_buffer=burst_buffer,
            comment=comment,
            core_count=core_count,
            duration=duration,
            end_time=end_time,
            features=features,
            flags=flags,
            groups=groups,
            licenses=licenses,
            max_start_delay=max_start_delay,
            name=name,
            node_count=node_count,
            node_list=node_list,
            partition=partition,
            purge_completed=purge_completed,
            start_time=start_time,
            tres=tres,
            users=users,
        )

        v0043_reservation_desc_msg.additional_properties = d
        return v0043_reservation_desc_msg

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
