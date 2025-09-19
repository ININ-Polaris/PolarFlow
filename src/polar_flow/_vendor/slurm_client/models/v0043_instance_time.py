from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043InstanceTime")


@_attrs_define
class V0043InstanceTime:
    """
    Attributes:
        time_end (Union[Unset, int]): When the instance will end (UNIX timestamp) (UNIX timestamp or time string
            recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
        time_start (Union[Unset, int]): When the instance will start (UNIX timestamp) (UNIX timestamp or time string
            recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
    """

    time_end: Union[Unset, int] = UNSET
    time_start: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        time_end = self.time_end

        time_start = self.time_start

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if time_end is not UNSET:
            field_dict["time_end"] = time_end
        if time_start is not UNSET:
            field_dict["time_start"] = time_start

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        time_end = d.pop("time_end", UNSET)

        time_start = d.pop("time_start", UNSET)

        v0043_instance_time = cls(
            time_end=time_end,
            time_start=time_start,
        )

        v0043_instance_time.additional_properties = d
        return v0043_instance_time

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
