from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043Tres")


@_attrs_define
class V0043Tres:
    """
    Attributes:
        type_ (str): TRES type (CPU, MEM, etc)
        name (Union[Unset, str]): TRES name (if applicable)
        id (Union[Unset, int]): ID used in the database
        count (Union[Unset, int]): TRES count (0 if listed generically)
    """

    type_: str
    name: Union[Unset, str] = UNSET
    id: Union[Unset, int] = UNSET
    count: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        name = self.name

        id = self.id

        count = self.count

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if id is not UNSET:
            field_dict["id"] = id
        if count is not UNSET:
            field_dict["count"] = count

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = d.pop("type")

        name = d.pop("name", UNSET)

        id = d.pop("id", UNSET)

        count = d.pop("count", UNSET)

        v0043_tres = cls(
            type_=type_,
            name=name,
            id=id,
            count=count,
        )

        v0043_tres.additional_properties = d
        return v0043_tres

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
