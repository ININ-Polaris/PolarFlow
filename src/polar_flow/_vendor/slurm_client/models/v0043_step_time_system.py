from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043StepTimeSystem")


@_attrs_define
class V0043StepTimeSystem:
    """
    Attributes:
        seconds (Union[Unset, int]): System CPU time used by the step in seconds
        microseconds (Union[Unset, int]): System CPU time used by the step in microseconds
    """

    seconds: Union[Unset, int] = UNSET
    microseconds: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        seconds = self.seconds

        microseconds = self.microseconds

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if seconds is not UNSET:
            field_dict["seconds"] = seconds
        if microseconds is not UNSET:
            field_dict["microseconds"] = microseconds

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        seconds = d.pop("seconds", UNSET)

        microseconds = d.pop("microseconds", UNSET)

        v0043_step_time_system = cls(
            seconds=seconds,
            microseconds=microseconds,
        )

        v0043_step_time_system.additional_properties = d
        return v0043_step_time_system

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
