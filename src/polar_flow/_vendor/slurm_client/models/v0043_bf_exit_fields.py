from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043BfExitFields")


@_attrs_define
class V0043BfExitFields:
    """
    Attributes:
        end_job_queue (Union[Unset, int]): Reached end of queue
        bf_max_job_start (Union[Unset, int]): Reached number of jobs allowed to start
        bf_max_job_test (Union[Unset, int]): Reached number of jobs allowed to be tested
        bf_max_time (Union[Unset, int]): Reached maximum allowed scheduler time
        bf_node_space_size (Union[Unset, int]): Reached table size limit
        state_changed (Union[Unset, int]): System state changed
    """

    end_job_queue: Union[Unset, int] = UNSET
    bf_max_job_start: Union[Unset, int] = UNSET
    bf_max_job_test: Union[Unset, int] = UNSET
    bf_max_time: Union[Unset, int] = UNSET
    bf_node_space_size: Union[Unset, int] = UNSET
    state_changed: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end_job_queue = self.end_job_queue

        bf_max_job_start = self.bf_max_job_start

        bf_max_job_test = self.bf_max_job_test

        bf_max_time = self.bf_max_time

        bf_node_space_size = self.bf_node_space_size

        state_changed = self.state_changed

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if end_job_queue is not UNSET:
            field_dict["end_job_queue"] = end_job_queue
        if bf_max_job_start is not UNSET:
            field_dict["bf_max_job_start"] = bf_max_job_start
        if bf_max_job_test is not UNSET:
            field_dict["bf_max_job_test"] = bf_max_job_test
        if bf_max_time is not UNSET:
            field_dict["bf_max_time"] = bf_max_time
        if bf_node_space_size is not UNSET:
            field_dict["bf_node_space_size"] = bf_node_space_size
        if state_changed is not UNSET:
            field_dict["state_changed"] = state_changed

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        end_job_queue = d.pop("end_job_queue", UNSET)

        bf_max_job_start = d.pop("bf_max_job_start", UNSET)

        bf_max_job_test = d.pop("bf_max_job_test", UNSET)

        bf_max_time = d.pop("bf_max_time", UNSET)

        bf_node_space_size = d.pop("bf_node_space_size", UNSET)

        state_changed = d.pop("state_changed", UNSET)

        v0043_bf_exit_fields = cls(
            end_job_queue=end_job_queue,
            bf_max_job_start=bf_max_job_start,
            bf_max_job_test=bf_max_job_test,
            bf_max_time=bf_max_time,
            bf_node_space_size=bf_node_space_size,
            state_changed=state_changed,
        )

        v0043_bf_exit_fields.additional_properties = d
        return v0043_bf_exit_fields

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
