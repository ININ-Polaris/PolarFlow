from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043PartitionInfoAccounts")


@_attrs_define
class V0043PartitionInfoAccounts:
    allowed: Union[Unset, str] = UNSET
    """ AllowAccounts - Comma-separated list of accounts which may execute jobs in the partition """
    deny: Union[Unset, str] = UNSET
    """ DenyAccounts - Comma-separated list of accounts which may not execute jobs in the partition """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        deny = self.deny

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allowed is not UNSET:
            field_dict["allowed"] = allowed
        if deny is not UNSET:
            field_dict["deny"] = deny

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        allowed = d.pop("allowed", UNSET)

        deny = d.pop("deny", UNSET)

        v0043_partition_info_accounts = cls(
            allowed=allowed,
            deny=deny,
        )

        v0043_partition_info_accounts.additional_properties = d
        return v0043_partition_info_accounts

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
