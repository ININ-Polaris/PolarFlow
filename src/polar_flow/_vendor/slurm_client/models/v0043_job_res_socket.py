from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.v0043_job_res_core import V0043JobResCore


T = TypeVar("T", bound="V0043JobResSocket")


@_attrs_define
class V0043JobResSocket:
    index: int
    """ Core index """
    cores: list["V0043JobResCore"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        cores = []
        for componentsschemasv0_0_43_job_res_core_array_item_data in self.cores:
            componentsschemasv0_0_43_job_res_core_array_item = (
                componentsschemasv0_0_43_job_res_core_array_item_data.to_dict()
            )
            cores.append(componentsschemasv0_0_43_job_res_core_array_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "cores": cores,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_res_core import V0043JobResCore

        d = dict(src_dict)
        index = d.pop("index")

        cores = []
        _cores = d.pop("cores")
        for componentsschemasv0_0_43_job_res_core_array_item_data in _cores:
            componentsschemasv0_0_43_job_res_core_array_item = V0043JobResCore.from_dict(
                componentsschemasv0_0_43_job_res_core_array_item_data
            )

            cores.append(componentsschemasv0_0_43_job_res_core_array_item)

        v0043_job_res_socket = cls(
            index=index,
            cores=cores,
        )

        v0043_job_res_socket.additional_properties = d
        return v0043_job_res_socket

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
