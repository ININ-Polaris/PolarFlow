from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_step_cpu_requested_frequency import V0043StepCPURequestedFrequency


T = TypeVar("T", bound="V0043StepCPU")


@_attrs_define
class V0043StepCPU:
    requested_frequency: Union[Unset, "V0043StepCPURequestedFrequency"] = UNSET
    governor: Union[Unset, str] = UNSET
    """ Requested CPU frequency governor in kHz """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        requested_frequency: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.requested_frequency, Unset):
            requested_frequency = self.requested_frequency.to_dict()

        governor = self.governor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if requested_frequency is not UNSET:
            field_dict["requested_frequency"] = requested_frequency
        if governor is not UNSET:
            field_dict["governor"] = governor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_step_cpu_requested_frequency import V0043StepCPURequestedFrequency

        d = dict(src_dict)
        _requested_frequency = d.pop("requested_frequency", UNSET)
        requested_frequency: Union[Unset, V0043StepCPURequestedFrequency]
        if isinstance(_requested_frequency, Unset):
            requested_frequency = UNSET
        else:
            requested_frequency = V0043StepCPURequestedFrequency.from_dict(_requested_frequency)

        governor = d.pop("governor", UNSET)

        v0043_step_cpu = cls(
            requested_frequency=requested_frequency,
            governor=governor,
        )

        v0043_step_cpu.additional_properties = d
        return v0043_step_cpu

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
