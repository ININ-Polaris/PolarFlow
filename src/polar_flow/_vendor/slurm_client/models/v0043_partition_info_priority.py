from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoPriority")


@_attrs_define
class V0043PartitionInfoPriority:
    """
    Attributes:
        job_factor (Union[Unset, int]): PriorityJobFactor - Partition factor used by priority/multifactor plugin in
            calculating job priority
        tier (Union[Unset, int]): PriorityTier - Controls the order in which the scheduler evaluates jobs from different
            partitions
    """

    job_factor: Union[Unset, int] = UNSET
    tier: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        job_factor = self.job_factor

        tier = self.tier

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if job_factor is not UNSET:
            field_dict["job_factor"] = job_factor
        if tier is not UNSET:
            field_dict["tier"] = tier

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        job_factor = d.pop("job_factor", UNSET)

        tier = d.pop("tier", UNSET)

        v0043_partition_info_priority = cls(
            job_factor=job_factor,
            tier=tier,
        )

        v0043_partition_info_priority.additional_properties = d
        return v0043_partition_info_priority

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
