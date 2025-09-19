from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_job_state_current_item import V0043JobStateCurrentItem
from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043JobState")


@_attrs_define
class V0043JobState:
    current: Union[Unset, list[V0043JobStateCurrentItem]] = UNSET
    """ Current state """
    reason: Union[Unset, str] = UNSET
    """ Reason for previous Pending or Failed state """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        current: Union[Unset, list[str]] = UNSET
        if not isinstance(self.current, Unset):
            current = []
            for current_item_data in self.current:
                current_item = current_item_data.value
                current.append(current_item)

        reason = self.reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if current is not UNSET:
            field_dict["current"] = current
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        current = []
        _current = d.pop("current", UNSET)
        for current_item_data in _current or []:
            current_item = V0043JobStateCurrentItem(current_item_data)

            current.append(current_item)

        reason = d.pop("reason", UNSET)

        v0043_job_state = cls(
            current=current,
            reason=reason,
        )

        v0043_job_state.additional_properties = d
        return v0043_job_state

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
