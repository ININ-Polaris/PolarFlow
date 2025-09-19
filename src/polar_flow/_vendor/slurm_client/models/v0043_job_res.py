from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_job_res_select_type_item import V0043JobResSelectTypeItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_res_nodes import V0043JobResNodes
    from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct


T = TypeVar("T", bound="V0043JobRes")


@_attrs_define
class V0043JobRes:
    """
    Attributes:
        select_type (list[V0043JobResSelectTypeItem]): Scheduler consumable resource selection type
        cpus (int): Number of allocated CPUs
        threads_per_core (V0043Uint16NoValStruct):
        nodes (Union[Unset, V0043JobResNodes]):
    """

    select_type: list[V0043JobResSelectTypeItem]
    cpus: int
    threads_per_core: "V0043Uint16NoValStruct"
    nodes: Union[Unset, "V0043JobResNodes"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        select_type = []
        for select_type_item_data in self.select_type:
            select_type_item = select_type_item_data.value
            select_type.append(select_type_item)

        cpus = self.cpus

        threads_per_core = self.threads_per_core.to_dict()

        nodes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.nodes, Unset):
            nodes = self.nodes.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "select_type": select_type,
                "cpus": cpus,
                "threads_per_core": threads_per_core,
            }
        )
        if nodes is not UNSET:
            field_dict["nodes"] = nodes

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_res_nodes import V0043JobResNodes
        from ..models.v0043_uint_16_no_val_struct import V0043Uint16NoValStruct

        d = dict(src_dict)
        select_type = []
        _select_type = d.pop("select_type")
        for select_type_item_data in _select_type:
            select_type_item = V0043JobResSelectTypeItem(select_type_item_data)

            select_type.append(select_type_item)

        cpus = d.pop("cpus")

        threads_per_core = V0043Uint16NoValStruct.from_dict(d.pop("threads_per_core"))

        _nodes = d.pop("nodes", UNSET)
        nodes: Union[Unset, V0043JobResNodes]
        if isinstance(_nodes, Unset):
            nodes = UNSET
        else:
            nodes = V0043JobResNodes.from_dict(_nodes)

        v0043_job_res = cls(
            select_type=select_type,
            cpus=cpus,
            threads_per_core=threads_per_core,
            nodes=nodes,
        )

        v0043_job_res.additional_properties = d
        return v0043_job_res

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
