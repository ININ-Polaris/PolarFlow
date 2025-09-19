from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043JobResNodeMemory")


@_attrs_define
class V0043JobResNodeMemory:
    """
    Attributes:
        used (Union[Unset, int]): Total memory (MiB) used by job
        allocated (Union[Unset, int]): Total memory (MiB) allocated to job
    """

    used: Union[Unset, int] = UNSET
    allocated: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        used = self.used

        allocated = self.allocated

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if used is not UNSET:
            field_dict["used"] = used
        if allocated is not UNSET:
            field_dict["allocated"] = allocated

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        used = d.pop("used", UNSET)

        allocated = d.pop("allocated", UNSET)

        v0043_job_res_node_memory = cls(
            used=used,
            allocated=allocated,
        )

        v0043_job_res_node_memory.additional_properties = d
        return v0043_job_res_node_memory

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
