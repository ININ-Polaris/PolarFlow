from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_desc_msg import V0043JobDescMsg


T = TypeVar("T", bound="V0043JobSubmitReq")


@_attrs_define
class V0043JobSubmitReq:
    """
    Attributes:
        script (Union[Unset, str]): Deprecated; Populate script field in jobs[0] or job
        jobs (Union[Unset, list['V0043JobDescMsg']]):
        job (Union[Unset, V0043JobDescMsg]):
    """

    script: Union[Unset, str] = UNSET
    jobs: Union[Unset, list["V0043JobDescMsg"]] = UNSET
    job: Union[Unset, "V0043JobDescMsg"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        script = self.script

        jobs: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.jobs, Unset):
            jobs = []
            for componentsschemasv0_0_43_job_desc_msg_list_item_data in self.jobs:
                componentsschemasv0_0_43_job_desc_msg_list_item = (
                    componentsschemasv0_0_43_job_desc_msg_list_item_data.to_dict()
                )
                jobs.append(componentsschemasv0_0_43_job_desc_msg_list_item)

        job: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.job, Unset):
            job = self.job.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if script is not UNSET:
            field_dict["script"] = script
        if jobs is not UNSET:
            field_dict["jobs"] = jobs
        if job is not UNSET:
            field_dict["job"] = job

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_desc_msg import V0043JobDescMsg

        d = dict(src_dict)
        script = d.pop("script", UNSET)

        jobs = []
        _jobs = d.pop("jobs", UNSET)
        for componentsschemasv0_0_43_job_desc_msg_list_item_data in _jobs or []:
            componentsschemasv0_0_43_job_desc_msg_list_item = V0043JobDescMsg.from_dict(
                componentsschemasv0_0_43_job_desc_msg_list_item_data
            )

            jobs.append(componentsschemasv0_0_43_job_desc_msg_list_item)

        _job = d.pop("job", UNSET)
        job: Union[Unset, V0043JobDescMsg]
        if isinstance(_job, Unset):
            job = UNSET
        else:
            job = V0043JobDescMsg.from_dict(_job)

        v0043_job_submit_req = cls(
            script=script,
            jobs=jobs,
            job=job,
        )

        v0043_job_submit_req.additional_properties = d
        return v0043_job_submit_req

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
