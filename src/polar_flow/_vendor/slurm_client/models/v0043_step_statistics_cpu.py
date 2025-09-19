from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043StepStatisticsCPU")


@_attrs_define
class V0043StepStatisticsCPU:
    actual_frequency: Union[Unset, int] = UNSET
    """ Average weighted CPU frequency of all tasks in kHz """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        actual_frequency = self.actual_frequency

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if actual_frequency is not UNSET:
            field_dict["actual_frequency"] = actual_frequency

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        actual_frequency = d.pop("actual_frequency", UNSET)

        v0043_step_statistics_cpu = cls(
            actual_frequency=actual_frequency,
        )

        v0043_step_statistics_cpu.additional_properties = d
        return v0043_step_statistics_cpu

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
