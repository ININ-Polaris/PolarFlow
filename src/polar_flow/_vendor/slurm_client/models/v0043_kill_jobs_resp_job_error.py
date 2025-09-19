from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043KillJobsRespJobError")


@_attrs_define
class V0043KillJobsRespJobError:
    """
    Attributes:
        string (Union[Unset, str]): String error encountered signaling job
        code (Union[Unset, int]): Numeric error encountered signaling job
        message (Union[Unset, str]): Error message why signaling job failed
    """

    string: Union[Unset, str] = UNSET
    code: Union[Unset, int] = UNSET
    message: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        string = self.string

        code = self.code

        message = self.message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if string is not UNSET:
            field_dict["string"] = string
        if code is not UNSET:
            field_dict["code"] = code
        if message is not UNSET:
            field_dict["message"] = message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        string = d.pop("string", UNSET)

        code = d.pop("code", UNSET)

        message = d.pop("message", UNSET)

        v0043_kill_jobs_resp_job_error = cls(
            string=string,
            code=code,
            message=message,
        )

        v0043_kill_jobs_resp_job_error.additional_properties = d
        return v0043_kill_jobs_resp_job_error

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
