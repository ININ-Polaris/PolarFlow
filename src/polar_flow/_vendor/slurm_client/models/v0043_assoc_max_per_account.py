from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043AssocMaxPerAccount")


@_attrs_define
class V0043AssocMaxPerAccount:
    """
    Attributes:
        wall_clock (Union[Unset, V0043Uint32NoValStruct]):
    """

    wall_clock: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        wall_clock: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.wall_clock, Unset):
            wall_clock = self.wall_clock.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if wall_clock is not UNSET:
            field_dict["wall_clock"] = wall_clock

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        _wall_clock = d.pop("wall_clock", UNSET)
        wall_clock: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_wall_clock, Unset):
            wall_clock = UNSET
        else:
            wall_clock = V0043Uint32NoValStruct.from_dict(_wall_clock)

        v0043_assoc_max_per_account = cls(
            wall_clock=wall_clock,
        )

        v0043_assoc_max_per_account.additional_properties = d
        return v0043_assoc_max_per_account

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
