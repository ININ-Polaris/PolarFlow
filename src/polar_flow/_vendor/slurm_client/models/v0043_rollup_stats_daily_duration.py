from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043RollupStatsDailyDuration")


@_attrs_define
class V0043RollupStatsDailyDuration:
    last: Union[Unset, int] = UNSET
    """ Total time spent doing daily daily rollup (seconds) """
    max_: Union[Unset, int] = UNSET
    """ Longest daily rollup time (seconds) """
    time: Union[Unset, int] = UNSET
    """ Total time spent doing daily rollups (seconds) """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        last = self.last

        max_ = self.max_

        time = self.time

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if last is not UNSET:
            field_dict["last"] = last
        if max_ is not UNSET:
            field_dict["max"] = max_
        if time is not UNSET:
            field_dict["time"] = time

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        last = d.pop("last", UNSET)

        max_ = d.pop("max", UNSET)

        time = d.pop("time", UNSET)

        v0043_rollup_stats_daily_duration = cls(
            last=last,
            max_=max_,
            time=time,
        )

        v0043_rollup_stats_daily_duration.additional_properties = d
        return v0043_rollup_stats_daily_duration

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
