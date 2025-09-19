from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct


T = TypeVar("T", bound="V0043ProcessExitCodeVerboseSignal")


@_attrs_define
class V0043ProcessExitCodeVerboseSignal:
    id: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    name: Union[Unset, str] = UNSET
    """ Signal sent to process (name) """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.id, Unset):
            id = self.id.to_dict()

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct

        d = dict(src_dict)
        _id = d.pop("id", UNSET)
        id: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_id, Unset):
            id = UNSET
        else:
            id = V0043Uint16NoValStruct.from_dict(_id)

        name = d.pop("name", UNSET)

        v0043_process_exit_code_verbose_signal = cls(
            id=id,
            name=name,
        )

        v0043_process_exit_code_verbose_signal.additional_properties = d
        return v0043_process_exit_code_verbose_signal

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
