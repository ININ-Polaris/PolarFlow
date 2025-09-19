from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_openapi_meta_slurm_version import V0043OpenapiMetaSlurmVersion


T = TypeVar("T", bound="V0043OpenapiMetaSlurm")


@_attrs_define
class V0043OpenapiMetaSlurm:
    version: Union[Unset, "V0043OpenapiMetaSlurmVersion"] = UNSET
    release: Union[Unset, str] = UNSET
    """ Slurm release string """
    cluster: Union[Unset, str] = UNSET
    """ Slurm cluster name """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        version: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.version, Unset):
            version = self.version.to_dict()

        release = self.release

        cluster = self.cluster

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if version is not UNSET:
            field_dict["version"] = version
        if release is not UNSET:
            field_dict["release"] = release
        if cluster is not UNSET:
            field_dict["cluster"] = cluster

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_openapi_meta_slurm_version import V0043OpenapiMetaSlurmVersion

        d = dict(src_dict)
        _version = d.pop("version", UNSET)
        version: Union[Unset, V0043OpenapiMetaSlurmVersion]
        if isinstance(_version, Unset):
            version = UNSET
        else:
            version = V0043OpenapiMetaSlurmVersion.from_dict(_version)

        release = d.pop("release", UNSET)

        cluster = d.pop("cluster", UNSET)

        v0043_openapi_meta_slurm = cls(
            version=version,
            release=release,
            cluster=cluster,
        )

        v0043_openapi_meta_slurm.additional_properties = d
        return v0043_openapi_meta_slurm

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
