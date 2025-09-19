from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043QosLimitsMaxJobsPer")


@_attrs_define
class V0043QosLimitsMaxJobsPer:
    account: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    user: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        account: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.account, Unset):
            account = self.account.to_dict()

        user: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.user, Unset):
            user = self.user.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if account is not UNSET:
            field_dict["account"] = account
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        _account = d.pop("account", UNSET)
        account: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_account, Unset):
            account = UNSET
        else:
            account = V0043Uint32NoValStruct.from_dict(_account)

        _user = d.pop("user", UNSET)
        user: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_user, Unset):
            user = UNSET
        else:
            user = V0043Uint32NoValStruct.from_dict(_user)

        v0043_qos_limits_max_jobs_per = cls(
            account=account,
            user=user,
        )

        v0043_qos_limits_max_jobs_per.additional_properties = d
        return v0043_qos_limits_max_jobs_per

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
