from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartPrio")


@_attrs_define
class V0043PartPrio:
    """
    Attributes:
        partition (Union[Unset, str]): Partition name
        priority (Union[Unset, int]): Prospective job priority if it runs in this partition
    """

    partition: Union[Unset, str] = UNSET
    priority: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        partition = self.partition

        priority = self.priority

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if partition is not UNSET:
            field_dict["partition"] = partition
        if priority is not UNSET:
            field_dict["priority"] = priority

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        partition = d.pop("partition", UNSET)

        priority = d.pop("priority", UNSET)

        v0043_part_prio = cls(
            partition=partition,
            priority=priority,
        )

        v0043_part_prio.additional_properties = d
        return v0043_part_prio

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
