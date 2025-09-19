from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_kill_jobs_resp_job_error import V0043KillJobsRespJobError
    from ..models.v0043_kill_jobs_resp_job_federation import V0043KillJobsRespJobFederation
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043KillJobsRespJob")


@_attrs_define
class V0043KillJobsRespJob:
    """
    Attributes:
        step_id (str): Job or Step ID that signaling failed
        job_id (V0043Uint32NoValStruct):
        error (Union[Unset, V0043KillJobsRespJobError]):
        federation (Union[Unset, V0043KillJobsRespJobFederation]):
    """

    step_id: str
    job_id: "V0043Uint32NoValStruct"
    error: Union[Unset, "V0043KillJobsRespJobError"] = UNSET
    federation: Union[Unset, "V0043KillJobsRespJobFederation"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        step_id = self.step_id

        job_id = self.job_id.to_dict()

        error: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.error, Unset):
            error = self.error.to_dict()

        federation: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.federation, Unset):
            federation = self.federation.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "step_id": step_id,
                "job_id": job_id,
            }
        )
        if error is not UNSET:
            field_dict["error"] = error
        if federation is not UNSET:
            field_dict["federation"] = federation

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_kill_jobs_resp_job_error import V0043KillJobsRespJobError
        from ..models.v0043_kill_jobs_resp_job_federation import V0043KillJobsRespJobFederation
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        step_id = d.pop("step_id")

        job_id = V0043Uint32NoValStruct.from_dict(d.pop("job_id"))

        _error = d.pop("error", UNSET)
        error: Union[Unset, V0043KillJobsRespJobError]
        if isinstance(_error, Unset):
            error = UNSET
        else:
            error = V0043KillJobsRespJobError.from_dict(_error)

        _federation = d.pop("federation", UNSET)
        federation: Union[Unset, V0043KillJobsRespJobFederation]
        if isinstance(_federation, Unset):
            federation = UNSET
        else:
            federation = V0043KillJobsRespJobFederation.from_dict(_federation)

        v0043_kill_jobs_resp_job = cls(
            step_id=step_id,
            job_id=job_id,
            error=error,
            federation=federation,
        )

        v0043_kill_jobs_resp_job.additional_properties = d
        return v0043_kill_jobs_resp_job

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
