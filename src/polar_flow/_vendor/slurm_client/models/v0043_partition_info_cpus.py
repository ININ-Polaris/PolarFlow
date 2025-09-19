from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoCpus")


@_attrs_define
class V0043PartitionInfoCpus:
    """
    Attributes:
        task_binding (Union[Unset, int]): CpuBind - Default method controlling how tasks are bound to allocated
            resources
        total (Union[Unset, int]): TotalCPUs - Number of CPUs available in this partition
    """

    task_binding: Union[Unset, int] = UNSET
    total: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_binding = self.task_binding

        total = self.total

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if task_binding is not UNSET:
            field_dict["task_binding"] = task_binding
        if total is not UNSET:
            field_dict["total"] = total

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task_binding = d.pop("task_binding", UNSET)

        total = d.pop("total", UNSET)

        v0043_partition_info_cpus = cls(
            task_binding=task_binding,
            total=total,
        )

        v0043_partition_info_cpus.additional_properties = d
        return v0043_partition_info_cpus

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
