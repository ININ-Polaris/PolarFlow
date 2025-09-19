from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_qos_limits_max_wall_clock_per import V0043QosLimitsMaxWallClockPer


T = TypeVar("T", bound="V0043QosLimitsMaxWallClock")


@_attrs_define
class V0043QosLimitsMaxWallClock:
    per: Union[Unset, "V0043QosLimitsMaxWallClockPer"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        per: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.per, Unset):
            per = self.per.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if per is not UNSET:
            field_dict["per"] = per

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_qos_limits_max_wall_clock_per import V0043QosLimitsMaxWallClockPer

        d = dict(src_dict)
        _per = d.pop("per", UNSET)
        per: Union[Unset, V0043QosLimitsMaxWallClockPer]
        if isinstance(_per, Unset):
            per = UNSET
        else:
            per = V0043QosLimitsMaxWallClockPer.from_dict(_per)

        v0043_qos_limits_max_wall_clock = cls(
            per=per,
        )

        v0043_qos_limits_max_wall_clock.additional_properties = d
        return v0043_qos_limits_max_wall_clock

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
