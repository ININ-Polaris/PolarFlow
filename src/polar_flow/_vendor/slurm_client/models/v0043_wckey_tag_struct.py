from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_wckey_tag_struct_flags_item import V0043WckeyTagStructFlagsItem

T = TypeVar("T", bound="V0043WckeyTagStruct")


@_attrs_define
class V0043WckeyTagStruct:
    wckey: str
    """ WCKey name """
    flags: list[V0043WckeyTagStructFlagsItem]
    """ Active flags """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        wckey = self.wckey

        flags = []
        for flags_item_data in self.flags:
            flags_item = flags_item_data.value
            flags.append(flags_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "wckey": wckey,
                "flags": flags,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        wckey = d.pop("wckey")

        flags = []
        _flags = d.pop("flags")
        for flags_item_data in _flags:
            flags_item = V0043WckeyTagStructFlagsItem(flags_item_data)

            flags.append(flags_item)

        v0043_wckey_tag_struct = cls(
            wckey=wckey,
            flags=flags,
        )

        v0043_wckey_tag_struct.additional_properties = d
        return v0043_wckey_tag_struct

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
