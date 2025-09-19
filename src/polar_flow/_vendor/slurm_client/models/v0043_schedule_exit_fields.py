from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043ScheduleExitFields")


@_attrs_define
class V0043ScheduleExitFields:
    """
    Attributes:
        end_job_queue (Union[Unset, int]): Reached end of queue
        default_queue_depth (Union[Unset, int]): Reached number of jobs allowed to be tested
        max_job_start (Union[Unset, int]): Reached number of jobs allowed to start
        max_rpc_cnt (Union[Unset, int]): Reached RPC limit
        max_sched_time (Union[Unset, int]): Reached maximum allowed scheduler time
        licenses (Union[Unset, int]): Blocked on licenses
    """

    end_job_queue: Union[Unset, int] = UNSET
    default_queue_depth: Union[Unset, int] = UNSET
    max_job_start: Union[Unset, int] = UNSET
    max_rpc_cnt: Union[Unset, int] = UNSET
    max_sched_time: Union[Unset, int] = UNSET
    licenses: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end_job_queue = self.end_job_queue

        default_queue_depth = self.default_queue_depth

        max_job_start = self.max_job_start

        max_rpc_cnt = self.max_rpc_cnt

        max_sched_time = self.max_sched_time

        licenses = self.licenses

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if end_job_queue is not UNSET:
            field_dict["end_job_queue"] = end_job_queue
        if default_queue_depth is not UNSET:
            field_dict["default_queue_depth"] = default_queue_depth
        if max_job_start is not UNSET:
            field_dict["max_job_start"] = max_job_start
        if max_rpc_cnt is not UNSET:
            field_dict["max_rpc_cnt"] = max_rpc_cnt
        if max_sched_time is not UNSET:
            field_dict["max_sched_time"] = max_sched_time
        if licenses is not UNSET:
            field_dict["licenses"] = licenses

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        end_job_queue = d.pop("end_job_queue", UNSET)

        default_queue_depth = d.pop("default_queue_depth", UNSET)

        max_job_start = d.pop("max_job_start", UNSET)

        max_rpc_cnt = d.pop("max_rpc_cnt", UNSET)

        max_sched_time = d.pop("max_sched_time", UNSET)

        licenses = d.pop("licenses", UNSET)

        v0043_schedule_exit_fields = cls(
            end_job_queue=end_job_queue,
            default_queue_depth=default_queue_depth,
            max_job_start=max_job_start,
            max_rpc_cnt=max_rpc_cnt,
            max_sched_time=max_sched_time,
            licenses=licenses,
        )

        v0043_schedule_exit_fields.additional_properties = d
        return v0043_schedule_exit_fields

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
