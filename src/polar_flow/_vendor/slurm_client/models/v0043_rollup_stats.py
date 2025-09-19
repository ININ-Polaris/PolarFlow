from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_rollup_stats_daily import V0043RollupStatsDaily
    from ..models.v0043_rollup_stats_hourly import V0043RollupStatsHourly
    from ..models.v0043_rollup_stats_monthly import V0043RollupStatsMonthly


T = TypeVar("T", bound="V0043RollupStats")


@_attrs_define
class V0043RollupStats:
    hourly: Union[Unset, "V0043RollupStatsHourly"] = UNSET
    daily: Union[Unset, "V0043RollupStatsDaily"] = UNSET
    monthly: Union[Unset, "V0043RollupStatsMonthly"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        hourly: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.hourly, Unset):
            hourly = self.hourly.to_dict()

        daily: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.daily, Unset):
            daily = self.daily.to_dict()

        monthly: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.monthly, Unset):
            monthly = self.monthly.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if hourly is not UNSET:
            field_dict["hourly"] = hourly
        if daily is not UNSET:
            field_dict["daily"] = daily
        if monthly is not UNSET:
            field_dict["monthly"] = monthly

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_rollup_stats_daily import V0043RollupStatsDaily
        from ..models.v0043_rollup_stats_hourly import V0043RollupStatsHourly
        from ..models.v0043_rollup_stats_monthly import V0043RollupStatsMonthly

        d = dict(src_dict)
        _hourly = d.pop("hourly", UNSET)
        hourly: Union[Unset, V0043RollupStatsHourly]
        if isinstance(_hourly, Unset):
            hourly = UNSET
        else:
            hourly = V0043RollupStatsHourly.from_dict(_hourly)

        _daily = d.pop("daily", UNSET)
        daily: Union[Unset, V0043RollupStatsDaily]
        if isinstance(_daily, Unset):
            daily = UNSET
        else:
            daily = V0043RollupStatsDaily.from_dict(_daily)

        _monthly = d.pop("monthly", UNSET)
        monthly: Union[Unset, V0043RollupStatsMonthly]
        if isinstance(_monthly, Unset):
            monthly = UNSET
        else:
            monthly = V0043RollupStatsMonthly.from_dict(_monthly)

        v0043_rollup_stats = cls(
            hourly=hourly,
            daily=daily,
            monthly=monthly,
        )

        v0043_rollup_stats.additional_properties = d
        return v0043_rollup_stats

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
