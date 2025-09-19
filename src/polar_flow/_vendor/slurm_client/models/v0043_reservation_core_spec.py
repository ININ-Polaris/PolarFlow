from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043ReservationCoreSpec")


@_attrs_define
class V0043ReservationCoreSpec:
    """
    Attributes:
        node (Union[Unset, str]): Name of reserved node
        core (Union[Unset, str]): IDs of reserved cores
    """

    node: Union[Unset, str] = UNSET
    core: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        node = self.node

        core = self.core

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if node is not UNSET:
            field_dict["node"] = node
        if core is not UNSET:
            field_dict["core"] = core

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        node = d.pop("node", UNSET)

        core = d.pop("core", UNSET)

        v0043_reservation_core_spec = cls(
            node=node,
            core=core,
        )

        v0043_reservation_core_spec.additional_properties = d
        return v0043_reservation_core_spec

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
