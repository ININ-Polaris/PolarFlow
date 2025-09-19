from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_float_64_no_val_struct import V0043Float64NoValStruct
    from ..models.v0043_qos_limits_max import V0043QosLimitsMax
    from ..models.v0043_qos_limits_min import V0043QosLimitsMin


T = TypeVar("T", bound="V0043QosLimits")


@_attrs_define
class V0043QosLimits:
    grace_time: Union[Unset, int] = UNSET
    """ GraceTime - Preemption grace time in seconds to be extended to a job which has been selected for preemption
    """
    max_: Union[Unset, "V0043QosLimitsMax"] = UNSET
    factor: Union[Unset, "V0043Float64NoValStruct"] = UNSET
    min_: Union[Unset, "V0043QosLimitsMin"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        grace_time = self.grace_time

        max_: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.max_, Unset):
            max_ = self.max_.to_dict()

        factor: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.factor, Unset):
            factor = self.factor.to_dict()

        min_: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.min_, Unset):
            min_ = self.min_.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if grace_time is not UNSET:
            field_dict["grace_time"] = grace_time
        if max_ is not UNSET:
            field_dict["max"] = max_
        if factor is not UNSET:
            field_dict["factor"] = factor
        if min_ is not UNSET:
            field_dict["min"] = min_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_float_64_no_val_struct import V0043Float64NoValStruct
        from ..models.v0043_qos_limits_max import V0043QosLimitsMax
        from ..models.v0043_qos_limits_min import V0043QosLimitsMin

        d = dict(src_dict)
        grace_time = d.pop("grace_time", UNSET)

        _max_ = d.pop("max", UNSET)
        max_: Union[Unset, V0043QosLimitsMax]
        if isinstance(_max_, Unset):
            max_ = UNSET
        else:
            max_ = V0043QosLimitsMax.from_dict(_max_)

        _factor = d.pop("factor", UNSET)
        factor: Union[Unset, V0043Float64NoValStruct]
        if isinstance(_factor, Unset):
            factor = UNSET
        else:
            factor = V0043Float64NoValStruct.from_dict(_factor)

        _min_ = d.pop("min", UNSET)
        min_: Union[Unset, V0043QosLimitsMin]
        if isinstance(_min_, Unset):
            min_ = UNSET
        else:
            min_ = V0043QosLimitsMin.from_dict(_min_)

        v0043_qos_limits = cls(
            grace_time=grace_time,
            max_=max_,
            factor=factor,
            min_=min_,
        )

        v0043_qos_limits.additional_properties = d
        return v0043_qos_limits

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
