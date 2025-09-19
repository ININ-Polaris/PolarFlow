from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_job_res_nodes_select_type_item import V0043JobResNodesSelectTypeItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_res_node import V0043JobResNode


T = TypeVar("T", bound="V0043JobResNodes")


@_attrs_define
class V0043JobResNodes:
    count: Union[Unset, int] = UNSET
    """ Number of allocated nodes """
    select_type: Union[Unset, list[V0043JobResNodesSelectTypeItem]] = UNSET
    """ Node scheduling selection method """
    list_: Union[Unset, str] = UNSET
    """ Node(s) allocated to the job """
    whole: Union[Unset, bool] = UNSET
    """ Whether whole nodes were allocated """
    allocation: Union[Unset, list["V0043JobResNode"]] = UNSET
    """ Job resources for a node """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        count = self.count

        select_type: Union[Unset, list[str]] = UNSET
        if not isinstance(self.select_type, Unset):
            select_type = []
            for select_type_item_data in self.select_type:
                select_type_item = select_type_item_data.value
                select_type.append(select_type_item)

        list_ = self.list_

        whole = self.whole

        allocation: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.allocation, Unset):
            allocation = []
            for componentsschemasv0_0_43_job_res_nodes_item_data in self.allocation:
                componentsschemasv0_0_43_job_res_nodes_item = componentsschemasv0_0_43_job_res_nodes_item_data.to_dict()
                allocation.append(componentsschemasv0_0_43_job_res_nodes_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if count is not UNSET:
            field_dict["count"] = count
        if select_type is not UNSET:
            field_dict["select_type"] = select_type
        if list_ is not UNSET:
            field_dict["list"] = list_
        if whole is not UNSET:
            field_dict["whole"] = whole
        if allocation is not UNSET:
            field_dict["allocation"] = allocation

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_res_node import V0043JobResNode

        d = dict(src_dict)
        count = d.pop("count", UNSET)

        select_type = []
        _select_type = d.pop("select_type", UNSET)
        for select_type_item_data in _select_type or []:
            select_type_item = V0043JobResNodesSelectTypeItem(select_type_item_data)

            select_type.append(select_type_item)

        list_ = d.pop("list", UNSET)

        whole = d.pop("whole", UNSET)

        allocation = []
        _allocation = d.pop("allocation", UNSET)
        for componentsschemasv0_0_43_job_res_nodes_item_data in _allocation or []:
            componentsschemasv0_0_43_job_res_nodes_item = V0043JobResNode.from_dict(
                componentsschemasv0_0_43_job_res_nodes_item_data
            )

            allocation.append(componentsschemasv0_0_43_job_res_nodes_item)

        v0043_job_res_nodes = cls(
            count=count,
            select_type=select_type,
            list_=list_,
            whole=whole,
            allocation=allocation,
        )

        v0043_job_res_nodes.additional_properties = d
        return v0043_job_res_nodes

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
