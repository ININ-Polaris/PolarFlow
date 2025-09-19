from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_assoc_max_per_account import V0043AssocMaxPerAccount


T = TypeVar("T", bound="V0043AssocMaxPer")


@_attrs_define
class V0043AssocMaxPer:
    account: Union[Unset, "V0043AssocMaxPerAccount"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        account: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.account, Unset):
            account = self.account.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if account is not UNSET:
            field_dict["account"] = account

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_assoc_max_per_account import V0043AssocMaxPerAccount

        d = dict(src_dict)
        _account = d.pop("account", UNSET)
        account: Union[Unset, V0043AssocMaxPerAccount]
        if isinstance(_account, Unset):
            account = UNSET
        else:
            account = V0043AssocMaxPerAccount.from_dict(_account)

        v0043_assoc_max_per = cls(
            account=account,
        )

        v0043_assoc_max_per.additional_properties = d
        return v0043_assoc_max_per

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
