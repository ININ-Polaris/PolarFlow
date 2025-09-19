from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043OpenapiMetaPlugin")


@_attrs_define
class V0043OpenapiMetaPlugin:
    type_: Union[Unset, str] = UNSET
    """ Slurm plugin type (if applicable) """
    name: Union[Unset, str] = UNSET
    """ Slurm plugin name (if applicable) """
    data_parser: Union[Unset, str] = UNSET
    """ Slurm data_parser plugin """
    accounting_storage: Union[Unset, str] = UNSET
    """ Slurm accounting plugin """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        name = self.name

        data_parser = self.data_parser

        accounting_storage = self.accounting_storage

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if name is not UNSET:
            field_dict["name"] = name
        if data_parser is not UNSET:
            field_dict["data_parser"] = data_parser
        if accounting_storage is not UNSET:
            field_dict["accounting_storage"] = accounting_storage

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = d.pop("type", UNSET)

        name = d.pop("name", UNSET)

        data_parser = d.pop("data_parser", UNSET)

        accounting_storage = d.pop("accounting_storage", UNSET)

        v0043_openapi_meta_plugin = cls(
            type_=type_,
            name=name,
            data_parser=data_parser,
            accounting_storage=accounting_storage,
        )

        v0043_openapi_meta_plugin.additional_properties = d
        return v0043_openapi_meta_plugin

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
