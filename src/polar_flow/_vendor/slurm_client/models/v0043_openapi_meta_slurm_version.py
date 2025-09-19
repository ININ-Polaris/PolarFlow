from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043OpenapiMetaSlurmVersion")


@_attrs_define
class V0043OpenapiMetaSlurmVersion:
    """
    Attributes:
        major (Union[Unset, str]): Slurm release major version
        micro (Union[Unset, str]): Slurm release micro version
        minor (Union[Unset, str]): Slurm release minor version
    """

    major: Union[Unset, str] = UNSET
    micro: Union[Unset, str] = UNSET
    minor: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        major = self.major

        micro = self.micro

        minor = self.minor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if major is not UNSET:
            field_dict["major"] = major
        if micro is not UNSET:
            field_dict["micro"] = micro
        if minor is not UNSET:
            field_dict["minor"] = minor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        major = d.pop("major", UNSET)

        micro = d.pop("micro", UNSET)

        minor = d.pop("minor", UNSET)

        v0043_openapi_meta_slurm_version = cls(
            major=major,
            micro=micro,
            minor=minor,
        )

        v0043_openapi_meta_slurm_version.additional_properties = d
        return v0043_openapi_meta_slurm_version

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
