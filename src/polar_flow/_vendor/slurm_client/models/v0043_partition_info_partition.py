from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_partition_info_partition_state_item import V0043PartitionInfoPartitionStateItem
from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoPartition")


@_attrs_define
class V0043PartitionInfoPartition:
    state: Union[Unset, list[V0043PartitionInfoPartitionStateItem]] = UNSET
    """ Current state(s) """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        state: Union[Unset, list[str]] = UNSET
        if not isinstance(self.state, Unset):
            state = []
            for state_item_data in self.state:
                state_item = state_item_data.value
                state.append(state_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if state is not UNSET:
            field_dict["state"] = state

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        state = []
        _state = d.pop("state", UNSET)
        for state_item_data in _state or []:
            state_item = V0043PartitionInfoPartitionStateItem(state_item_data)

            state.append(state_item)

        v0043_partition_info_partition = cls(
            state=state,
        )

        v0043_partition_info_partition.additional_properties = d
        return v0043_partition_info_partition

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
