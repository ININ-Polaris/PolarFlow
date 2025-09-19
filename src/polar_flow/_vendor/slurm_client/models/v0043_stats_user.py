from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_stats_user_time import V0043StatsUserTime


T = TypeVar("T", bound="V0043StatsUser")


@_attrs_define
class V0043StatsUser:
    """
    Attributes:
        user (Union[Unset, str]): User ID
        count (Union[Unset, int]): Number of RPCs processed
        time (Union[Unset, V0043StatsUserTime]):
    """

    user: Union[Unset, str] = UNSET
    count: Union[Unset, int] = UNSET
    time: Union[Unset, "V0043StatsUserTime"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user = self.user

        count = self.count

        time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.time, Unset):
            time = self.time.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if user is not UNSET:
            field_dict["user"] = user
        if count is not UNSET:
            field_dict["count"] = count
        if time is not UNSET:
            field_dict["time"] = time

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_stats_user_time import V0043StatsUserTime

        d = dict(src_dict)
        user = d.pop("user", UNSET)

        count = d.pop("count", UNSET)

        _time = d.pop("time", UNSET)
        time: Union[Unset, V0043StatsUserTime]
        if isinstance(_time, Unset):
            time = UNSET
        else:
            time = V0043StatsUserTime.from_dict(_time)

        v0043_stats_user = cls(
            user=user,
            count=count,
            time=time,
        )

        v0043_stats_user.additional_properties = d
        return v0043_stats_user

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
