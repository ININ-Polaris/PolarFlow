from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043Uint32NoValStruct")


@_attrs_define
class V0043Uint32NoValStruct:
    set_: Union[Unset, bool] = UNSET
    """ True if number has been set; False if number is unset """
    infinite: Union[Unset, bool] = UNSET
    """ True if number has been set to infinite; "set" and "number" will be ignored """
    number: Union[Unset, int] = UNSET
    """ If "set" is True the number will be set with value; otherwise ignore number contents """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        set_ = self.set_

        infinite = self.infinite

        number = self.number

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if set_ is not UNSET:
            field_dict["set"] = set_
        if infinite is not UNSET:
            field_dict["infinite"] = infinite
        if number is not UNSET:
            field_dict["number"] = number

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        set_ = d.pop("set", UNSET)

        infinite = d.pop("infinite", UNSET)

        number = d.pop("number", UNSET)

        v0043_uint_32_no_val_struct = cls(
            set_=set_,
            infinite=infinite,
            number=number,
        )

        v0043_uint_32_no_val_struct.additional_properties = d
        return v0043_uint_32_no_val_struct

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
