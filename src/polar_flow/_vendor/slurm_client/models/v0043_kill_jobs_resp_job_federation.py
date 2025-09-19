from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043KillJobsRespJobFederation")


@_attrs_define
class V0043KillJobsRespJobFederation:
    """
    Attributes:
        sibling (Union[Unset, str]): Name of federation sibling (may be empty for non-federation)
    """

    sibling: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sibling = self.sibling

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if sibling is not UNSET:
            field_dict["sibling"] = sibling

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        sibling = d.pop("sibling", UNSET)

        v0043_kill_jobs_resp_job_federation = cls(
            sibling=sibling,
        )

        v0043_kill_jobs_resp_job_federation.additional_properties = d
        return v0043_kill_jobs_resp_job_federation

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
