from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043OpenapiError")


@_attrs_define
class V0043OpenapiError:
    """
    Attributes:
        description (Union[Unset, str]): Long form error description
        error_number (Union[Unset, int]): Slurm numeric error identifier
        error (Union[Unset, str]): Short form error description
        source (Union[Unset, str]): Source of error or where error was first detected
    """

    description: Union[Unset, str] = UNSET
    error_number: Union[Unset, int] = UNSET
    error: Union[Unset, str] = UNSET
    source: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        error_number = self.error_number

        error = self.error

        source = self.source

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if description is not UNSET:
            field_dict["description"] = description
        if error_number is not UNSET:
            field_dict["error_number"] = error_number
        if error is not UNSET:
            field_dict["error"] = error
        if source is not UNSET:
            field_dict["source"] = source

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        description = d.pop("description", UNSET)

        error_number = d.pop("error_number", UNSET)

        error = d.pop("error", UNSET)

        source = d.pop("source", UNSET)

        v0043_openapi_error = cls(
            description=description,
            error_number=error_number,
            error=error,
            source=source,
        )

        v0043_openapi_error.additional_properties = d
        return v0043_openapi_error

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
