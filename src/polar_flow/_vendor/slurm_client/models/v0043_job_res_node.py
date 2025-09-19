from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_res_node_cpus import V0043JobResNodeCpus
    from ..models.v0043_job_res_node_memory import V0043JobResNodeMemory
    from ..models.v0043_job_res_socket import V0043JobResSocket


T = TypeVar("T", bound="V0043JobResNode")


@_attrs_define
class V0043JobResNode:
    index: int
    """ Node index """
    name: str
    """ Node name """
    sockets: list["V0043JobResSocket"]
    cpus: Union[Unset, "V0043JobResNodeCpus"] = UNSET
    memory: Union[Unset, "V0043JobResNodeMemory"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        name = self.name

        sockets = []
        for componentsschemasv0_0_43_job_res_socket_array_item_data in self.sockets:
            componentsschemasv0_0_43_job_res_socket_array_item = (
                componentsschemasv0_0_43_job_res_socket_array_item_data.to_dict()
            )
            sockets.append(componentsschemasv0_0_43_job_res_socket_array_item)

        cpus: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.cpus, Unset):
            cpus = self.cpus.to_dict()

        memory: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.memory, Unset):
            memory = self.memory.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "name": name,
                "sockets": sockets,
            }
        )
        if cpus is not UNSET:
            field_dict["cpus"] = cpus
        if memory is not UNSET:
            field_dict["memory"] = memory

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_res_node_cpus import V0043JobResNodeCpus
        from ..models.v0043_job_res_node_memory import V0043JobResNodeMemory
        from ..models.v0043_job_res_socket import V0043JobResSocket

        d = dict(src_dict)
        index = d.pop("index")

        name = d.pop("name")

        sockets = []
        _sockets = d.pop("sockets")
        for componentsschemasv0_0_43_job_res_socket_array_item_data in _sockets:
            componentsschemasv0_0_43_job_res_socket_array_item = V0043JobResSocket.from_dict(
                componentsschemasv0_0_43_job_res_socket_array_item_data
            )

            sockets.append(componentsschemasv0_0_43_job_res_socket_array_item)

        _cpus = d.pop("cpus", UNSET)
        cpus: Union[Unset, V0043JobResNodeCpus]
        if isinstance(_cpus, Unset):
            cpus = UNSET
        else:
            cpus = V0043JobResNodeCpus.from_dict(_cpus)

        _memory = d.pop("memory", UNSET)
        memory: Union[Unset, V0043JobResNodeMemory]
        if isinstance(_memory, Unset):
            memory = UNSET
        else:
            memory = V0043JobResNodeMemory.from_dict(_memory)

        v0043_job_res_node = cls(
            index=index,
            name=name,
            sockets=sockets,
            cpus=cpus,
            memory=memory,
        )

        v0043_job_res_node.additional_properties = d
        return v0043_job_res_node

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
