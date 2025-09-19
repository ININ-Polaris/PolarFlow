from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoQos")


@_attrs_define
class V0043PartitionInfoQos:
    allowed: Union[Unset, str] = UNSET
    """ AllowQOS - Comma-separated list of Qos which may execute jobs in the partition """
    deny: Union[Unset, str] = UNSET
    """ DenyQOS - Comma-separated list of Qos which may not execute jobs in the partition """
    assigned: Union[Unset, str] = UNSET
    """ QOS - QOS name containing limits that will apply to all jobs in this partition """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        deny = self.deny

        assigned = self.assigned

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allowed is not UNSET:
            field_dict["allowed"] = allowed
        if deny is not UNSET:
            field_dict["deny"] = deny
        if assigned is not UNSET:
            field_dict["assigned"] = assigned

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        allowed = d.pop("allowed", UNSET)

        deny = d.pop("deny", UNSET)

        assigned = d.pop("assigned", UNSET)

        v0043_partition_info_qos = cls(
            allowed=allowed,
            deny=deny,
            assigned=assigned,
        )

        v0043_partition_info_qos.additional_properties = d
        return v0043_partition_info_qos

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
