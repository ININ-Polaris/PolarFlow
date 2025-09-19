from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043JobComment")


@_attrs_define
class V0043JobComment:
    """
    Attributes:
        administrator (Union[Unset, str]): Arbitrary comment made by administrator
        job (Union[Unset, str]): Arbitrary comment made by user
        system (Union[Unset, str]): Arbitrary comment from slurmctld
    """

    administrator: Union[Unset, str] = UNSET
    job: Union[Unset, str] = UNSET
    system: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        administrator = self.administrator

        job = self.job

        system = self.system

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if administrator is not UNSET:
            field_dict["administrator"] = administrator
        if job is not UNSET:
            field_dict["job"] = job
        if system is not UNSET:
            field_dict["system"] = system

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        administrator = d.pop("administrator", UNSET)

        job = d.pop("job", UNSET)

        system = d.pop("system", UNSET)

        v0043_job_comment = cls(
            administrator=administrator,
            job=job,
            system=system,
        )

        v0043_job_comment.additional_properties = d
        return v0043_job_comment

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
