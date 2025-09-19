from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043StatsMsgRpcType")


@_attrs_define
class V0043StatsMsgRpcType:
    """
    Attributes:
        type_id (int): Message type as integer
        message_type (str): Message type as string (Slurm RPC message type)
        count (int): Number of RPCs received
        queued (int): Number of RPCs queued
        dropped (int): Number of RPCs dropped
        cycle_last (int): Number of RPCs processed within the last RPC queue cycle
        cycle_max (int): Maximum number of RPCs processed within a RPC queue cycle since start
        total_time (int): Total time spent processing RPC in seconds
        average_time (V0043Uint64NoValStruct):
    """

    type_id: int
    message_type: str
    count: int
    queued: int
    dropped: int
    cycle_last: int
    cycle_max: int
    total_time: int
    average_time: "V0043Uint64NoValStruct"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_id = self.type_id

        message_type = self.message_type

        count = self.count

        queued = self.queued

        dropped = self.dropped

        cycle_last = self.cycle_last

        cycle_max = self.cycle_max

        total_time = self.total_time

        average_time = self.average_time.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type_id": type_id,
                "message_type": message_type,
                "count": count,
                "queued": queued,
                "dropped": dropped,
                "cycle_last": cycle_last,
                "cycle_max": cycle_max,
                "total_time": total_time,
                "average_time": average_time,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        type_id = d.pop("type_id")

        message_type = d.pop("message_type")

        count = d.pop("count")

        queued = d.pop("queued")

        dropped = d.pop("dropped")

        cycle_last = d.pop("cycle_last")

        cycle_max = d.pop("cycle_max")

        total_time = d.pop("total_time")

        average_time = V0043Uint64NoValStruct.from_dict(d.pop("average_time"))

        v0043_stats_msg_rpc_type = cls(
            type_id=type_id,
            message_type=message_type,
            count=count,
            queued=queued,
            dropped=dropped,
            cycle_last=cycle_last,
            cycle_max=cycle_max,
            total_time=total_time,
            average_time=average_time,
        )

        v0043_stats_msg_rpc_type.additional_properties = d
        return v0043_stats_msg_rpc_type

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
