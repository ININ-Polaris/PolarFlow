from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_array_limits_max import V0043JobArrayLimitsMax


T = TypeVar("T", bound="V0043JobArrayLimits")


@_attrs_define
class V0043JobArrayLimits:
    max_: Union[Unset, "V0043JobArrayLimitsMax"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        max_: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.max_, Unset):
            max_ = self.max_.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if max_ is not UNSET:
            field_dict["max"] = max_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_array_limits_max import V0043JobArrayLimitsMax

        d = dict(src_dict)
        _max_ = d.pop("max", UNSET)
        max_: Union[Unset, V0043JobArrayLimitsMax]
        if isinstance(_max_, Unset):
            max_ = UNSET
        else:
            max_ = V0043JobArrayLimitsMax.from_dict(_max_)

        v0043_job_array_limits = cls(
            max_=max_,
        )

        v0043_job_array_limits.additional_properties = d
        return v0043_job_array_limits

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
