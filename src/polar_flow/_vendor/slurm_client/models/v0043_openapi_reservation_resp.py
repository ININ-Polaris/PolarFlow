from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_openapi_error import V0043OpenapiError
    from ..models.v0043_openapi_meta import V0043OpenapiMeta
    from ..models.v0043_openapi_warning import V0043OpenapiWarning
    from ..models.v0043_reservation_info import V0043ReservationInfo
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043OpenapiReservationResp")


@_attrs_define
class V0043OpenapiReservationResp:
    """
    Attributes:
        reservations (list['V0043ReservationInfo']):
        last_update (V0043Uint64NoValStruct):
        meta (Union[Unset, V0043OpenapiMeta]):
        errors (Union[Unset, list['V0043OpenapiError']]):
        warnings (Union[Unset, list['V0043OpenapiWarning']]):
    """

    reservations: list["V0043ReservationInfo"]
    last_update: "V0043Uint64NoValStruct"
    meta: Union[Unset, "V0043OpenapiMeta"] = UNSET
    errors: Union[Unset, list["V0043OpenapiError"]] = UNSET
    warnings: Union[Unset, list["V0043OpenapiWarning"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reservations = []
        for componentsschemasv0_0_43_reservation_info_msg_item_data in self.reservations:
            componentsschemasv0_0_43_reservation_info_msg_item = (
                componentsschemasv0_0_43_reservation_info_msg_item_data.to_dict()
            )
            reservations.append(componentsschemasv0_0_43_reservation_info_msg_item)

        last_update = self.last_update.to_dict()

        meta: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.meta, Unset):
            meta = self.meta.to_dict()

        errors: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.errors, Unset):
            errors = []
            for componentsschemasv0_0_43_openapi_errors_item_data in self.errors:
                componentsschemasv0_0_43_openapi_errors_item = (
                    componentsschemasv0_0_43_openapi_errors_item_data.to_dict()
                )
                errors.append(componentsschemasv0_0_43_openapi_errors_item)

        warnings: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.warnings, Unset):
            warnings = []
            for componentsschemasv0_0_43_openapi_warnings_item_data in self.warnings:
                componentsschemasv0_0_43_openapi_warnings_item = (
                    componentsschemasv0_0_43_openapi_warnings_item_data.to_dict()
                )
                warnings.append(componentsschemasv0_0_43_openapi_warnings_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "reservations": reservations,
                "last_update": last_update,
            }
        )
        if meta is not UNSET:
            field_dict["meta"] = meta
        if errors is not UNSET:
            field_dict["errors"] = errors
        if warnings is not UNSET:
            field_dict["warnings"] = warnings

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_openapi_error import V0043OpenapiError
        from ..models.v0043_openapi_meta import V0043OpenapiMeta
        from ..models.v0043_openapi_warning import V0043OpenapiWarning
        from ..models.v0043_reservation_info import V0043ReservationInfo
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        reservations = []
        _reservations = d.pop("reservations")
        for componentsschemasv0_0_43_reservation_info_msg_item_data in _reservations:
            componentsschemasv0_0_43_reservation_info_msg_item = V0043ReservationInfo.from_dict(
                componentsschemasv0_0_43_reservation_info_msg_item_data
            )

            reservations.append(componentsschemasv0_0_43_reservation_info_msg_item)

        last_update = V0043Uint64NoValStruct.from_dict(d.pop("last_update"))

        _meta = d.pop("meta", UNSET)
        meta: Union[Unset, V0043OpenapiMeta]
        if isinstance(_meta, Unset):
            meta = UNSET
        else:
            meta = V0043OpenapiMeta.from_dict(_meta)

        errors = []
        _errors = d.pop("errors", UNSET)
        for componentsschemasv0_0_43_openapi_errors_item_data in _errors or []:
            componentsschemasv0_0_43_openapi_errors_item = V0043OpenapiError.from_dict(
                componentsschemasv0_0_43_openapi_errors_item_data
            )

            errors.append(componentsschemasv0_0_43_openapi_errors_item)

        warnings = []
        _warnings = d.pop("warnings", UNSET)
        for componentsschemasv0_0_43_openapi_warnings_item_data in _warnings or []:
            componentsschemasv0_0_43_openapi_warnings_item = V0043OpenapiWarning.from_dict(
                componentsschemasv0_0_43_openapi_warnings_item_data
            )

            warnings.append(componentsschemasv0_0_43_openapi_warnings_item)

        v0043_openapi_reservation_resp = cls(
            reservations=reservations,
            last_update=last_update,
            meta=meta,
            errors=errors,
            warnings=warnings,
        )

        v0043_openapi_reservation_resp.additional_properties = d
        return v0043_openapi_reservation_resp

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
