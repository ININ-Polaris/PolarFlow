from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_tres import V0043Tres


T = TypeVar("T", bound="V0043JobTres")


@_attrs_define
class V0043JobTres:
    """
    Attributes:
        allocated (Union[Unset, list['V0043Tres']]):
        requested (Union[Unset, list['V0043Tres']]):
    """

    allocated: Union[Unset, list["V0043Tres"]] = UNSET
    requested: Union[Unset, list["V0043Tres"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allocated: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.allocated, Unset):
            allocated = []
            for componentsschemasv0_0_43_tres_list_item_data in self.allocated:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                allocated.append(componentsschemasv0_0_43_tres_list_item)

        requested: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.requested, Unset):
            requested = []
            for componentsschemasv0_0_43_tres_list_item_data in self.requested:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                requested.append(componentsschemasv0_0_43_tres_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allocated is not UNSET:
            field_dict["allocated"] = allocated
        if requested is not UNSET:
            field_dict["requested"] = requested

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_tres import V0043Tres

        d = dict(src_dict)
        allocated = []
        _allocated = d.pop("allocated", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _allocated or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            allocated.append(componentsschemasv0_0_43_tres_list_item)

        requested = []
        _requested = d.pop("requested", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _requested or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            requested.append(componentsschemasv0_0_43_tres_list_item)

        v0043_job_tres = cls(
            allocated=allocated,
            requested=requested,
        )

        v0043_job_tres.additional_properties = d
        return v0043_job_tres

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
