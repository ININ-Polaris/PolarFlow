from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_job_res_core_status_item import V0043JobResCoreStatusItem

T = TypeVar("T", bound="V0043JobResCore")


@_attrs_define
class V0043JobResCore:
    """
    Attributes:
        index (int): Core index
        status (list[V0043JobResCoreStatusItem]): Core status
    """

    index: int
    status: list[V0043JobResCoreStatusItem]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        status = []
        for status_item_data in self.status:
            status_item = status_item_data.value
            status.append(status_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "status": status,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        index = d.pop("index")

        status = []
        _status = d.pop("status")
        for status_item_data in _status:
            status_item = V0043JobResCoreStatusItem(status_item_data)

            status.append(status_item)

        v0043_job_res_core = cls(
            index=index,
            status=status,
        )

        v0043_job_res_core.additional_properties = d
        return v0043_job_res_core

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
