from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_reservation_desc_msg import V0043ReservationDescMsg


T = TypeVar("T", bound="V0043ReservationModReq")


@_attrs_define
class V0043ReservationModReq:
    """
    Attributes:
        reservations (Union[Unset, list['V0043ReservationDescMsg']]):
    """

    reservations: Union[Unset, list["V0043ReservationDescMsg"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reservations: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.reservations, Unset):
            reservations = []
            for componentsschemasv0_0_43_reservation_desc_msg_list_item_data in self.reservations:
                componentsschemasv0_0_43_reservation_desc_msg_list_item = (
                    componentsschemasv0_0_43_reservation_desc_msg_list_item_data.to_dict()
                )
                reservations.append(componentsschemasv0_0_43_reservation_desc_msg_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if reservations is not UNSET:
            field_dict["reservations"] = reservations

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_reservation_desc_msg import V0043ReservationDescMsg

        d = dict(src_dict)
        reservations = []
        _reservations = d.pop("reservations", UNSET)
        for componentsschemasv0_0_43_reservation_desc_msg_list_item_data in _reservations or []:
            componentsschemasv0_0_43_reservation_desc_msg_list_item = V0043ReservationDescMsg.from_dict(
                componentsschemasv0_0_43_reservation_desc_msg_list_item_data
            )

            reservations.append(componentsschemasv0_0_43_reservation_desc_msg_list_item)

        v0043_reservation_mod_req = cls(
            reservations=reservations,
        )

        v0043_reservation_mod_req.additional_properties = d
        return v0043_reservation_mod_req

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
