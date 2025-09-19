from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoNodes")


@_attrs_define
class V0043PartitionInfoNodes:
    allowed_allocation: Union[Unset, str] = UNSET
    """ AllocNodes - Comma-separated list of nodes from which users can submit jobs in the partition """
    configured: Union[Unset, str] = UNSET
    """ Nodes - Comma-separated list of nodes which are associated with this partition """
    total: Union[Unset, int] = UNSET
    """ TotalNodes - Number of nodes available in this partition """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed_allocation = self.allowed_allocation

        configured = self.configured

        total = self.total

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allowed_allocation is not UNSET:
            field_dict["allowed_allocation"] = allowed_allocation
        if configured is not UNSET:
            field_dict["configured"] = configured
        if total is not UNSET:
            field_dict["total"] = total

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        allowed_allocation = d.pop("allowed_allocation", UNSET)

        configured = d.pop("configured", UNSET)

        total = d.pop("total", UNSET)

        v0043_partition_info_nodes = cls(
            allowed_allocation=allowed_allocation,
            configured=configured,
            total=total,
        )

        v0043_partition_info_nodes.additional_properties = d
        return v0043_partition_info_nodes

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
