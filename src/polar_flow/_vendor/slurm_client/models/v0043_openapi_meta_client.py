from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043OpenapiMetaClient")


@_attrs_define
class V0043OpenapiMetaClient:
    """
    Attributes:
        source (Union[Unset, str]): Client source description
        user (Union[Unset, str]): Client user (if known)
        group (Union[Unset, str]): Client group (if known)
    """

    source: Union[Unset, str] = UNSET
    user: Union[Unset, str] = UNSET
    group: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source = self.source

        user = self.user

        group = self.group

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if source is not UNSET:
            field_dict["source"] = source
        if user is not UNSET:
            field_dict["user"] = user
        if group is not UNSET:
            field_dict["group"] = group

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        source = d.pop("source", UNSET)

        user = d.pop("user", UNSET)

        group = d.pop("group", UNSET)

        v0043_openapi_meta_client = cls(
            source=source,
            user=user,
            group=group,
        )

        v0043_openapi_meta_client.additional_properties = d
        return v0043_openapi_meta_client

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
