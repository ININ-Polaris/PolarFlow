from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_tres import V0043Tres


T = TypeVar("T", bound="V0043AssocMaxTresGroup")


@_attrs_define
class V0043AssocMaxTresGroup:
    """
    Attributes:
        minutes (Union[Unset, list['V0043Tres']]):
        active (Union[Unset, list['V0043Tres']]):
    """

    minutes: Union[Unset, list["V0043Tres"]] = UNSET
    active: Union[Unset, list["V0043Tres"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        minutes: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.minutes, Unset):
            minutes = []
            for componentsschemasv0_0_43_tres_list_item_data in self.minutes:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                minutes.append(componentsschemasv0_0_43_tres_list_item)

        active: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.active, Unset):
            active = []
            for componentsschemasv0_0_43_tres_list_item_data in self.active:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                active.append(componentsschemasv0_0_43_tres_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if minutes is not UNSET:
            field_dict["minutes"] = minutes
        if active is not UNSET:
            field_dict["active"] = active

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_tres import V0043Tres

        d = dict(src_dict)
        minutes = []
        _minutes = d.pop("minutes", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _minutes or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            minutes.append(componentsschemasv0_0_43_tres_list_item)

        active = []
        _active = d.pop("active", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _active or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            active.append(componentsschemasv0_0_43_tres_list_item)

        v0043_assoc_max_tres_group = cls(
            minutes=minutes,
            active=active,
        )

        v0043_assoc_max_tres_group.additional_properties = d
        return v0043_assoc_max_tres_group

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
