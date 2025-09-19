from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043JobReservation")


@_attrs_define
class V0043JobReservation:
    """
    Attributes:
        id (Union[Unset, int]): Unique identifier of requested reservation
        name (Union[Unset, str]): Name of reservation to use
        requested (Union[Unset, str]): Comma-separated list of requested reservation names
    """

    id: Union[Unset, int] = UNSET
    name: Union[Unset, str] = UNSET
    requested: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        requested = self.requested

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name
        if requested is not UNSET:
            field_dict["requested"] = requested

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        requested = d.pop("requested", UNSET)

        v0043_job_reservation = cls(
            id=id,
            name=name,
            requested=requested,
        )

        v0043_job_reservation.additional_properties = d
        return v0043_job_reservation

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
