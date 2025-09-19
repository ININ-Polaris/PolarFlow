from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct


T = TypeVar("T", bound="V0043PartitionInfoTimeouts")


@_attrs_define
class V0043PartitionInfoTimeouts:
    resume: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    suspend: Union[Unset, "V0043Uint16NoValStruct"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        resume: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.resume, Unset):
            resume = self.resume.to_dict()

        suspend: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.suspend, Unset):
            suspend = self.suspend.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if resume is not UNSET:
            field_dict["resume"] = resume
        if suspend is not UNSET:
            field_dict["suspend"] = suspend

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct

        d = dict(src_dict)
        _resume = d.pop("resume", UNSET)
        resume: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_resume, Unset):
            resume = UNSET
        else:
            resume = V0043Uint16NoValStruct.from_dict(_resume)

        _suspend = d.pop("suspend", UNSET)
        suspend: Union[Unset, V0043Uint16NoValStruct]
        if isinstance(_suspend, Unset):
            suspend = UNSET
        else:
            suspend = V0043Uint16NoValStruct.from_dict(_suspend)

        v0043_partition_info_timeouts = cls(
            resume=resume,
            suspend=suspend,
        )

        v0043_partition_info_timeouts.additional_properties = d
        return v0043_partition_info_timeouts

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
