from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_shares_float_128_tres import V0043SharesFloat128Tres
    from ..models.v0043_shares_uint_64_tres import V0043SharesUint64Tres


T = TypeVar("T", bound="V0043AssocSharesObjWrapTres")


@_attrs_define
class V0043AssocSharesObjWrapTres:
    run_seconds: Union[Unset, list["V0043SharesUint64Tres"]] = UNSET
    group_minutes: Union[Unset, list["V0043SharesUint64Tres"]] = UNSET
    usage: Union[Unset, list["V0043SharesFloat128Tres"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        run_seconds: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.run_seconds, Unset):
            run_seconds = []
            for componentsschemasv0_0_43_shares_uint64_tres_list_item_data in self.run_seconds:
                componentsschemasv0_0_43_shares_uint64_tres_list_item = (
                    componentsschemasv0_0_43_shares_uint64_tres_list_item_data.to_dict()
                )
                run_seconds.append(componentsschemasv0_0_43_shares_uint64_tres_list_item)

        group_minutes: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.group_minutes, Unset):
            group_minutes = []
            for componentsschemasv0_0_43_shares_uint64_tres_list_item_data in self.group_minutes:
                componentsschemasv0_0_43_shares_uint64_tres_list_item = (
                    componentsschemasv0_0_43_shares_uint64_tres_list_item_data.to_dict()
                )
                group_minutes.append(componentsschemasv0_0_43_shares_uint64_tres_list_item)

        usage: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.usage, Unset):
            usage = []
            for componentsschemasv0_0_43_shares_float128_tres_list_item_data in self.usage:
                componentsschemasv0_0_43_shares_float128_tres_list_item = (
                    componentsschemasv0_0_43_shares_float128_tres_list_item_data.to_dict()
                )
                usage.append(componentsschemasv0_0_43_shares_float128_tres_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if run_seconds is not UNSET:
            field_dict["run_seconds"] = run_seconds
        if group_minutes is not UNSET:
            field_dict["group_minutes"] = group_minutes
        if usage is not UNSET:
            field_dict["usage"] = usage

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_shares_float_128_tres import V0043SharesFloat128Tres
        from ..models.v0043_shares_uint_64_tres import V0043SharesUint64Tres

        d = dict(src_dict)
        run_seconds = []
        _run_seconds = d.pop("run_seconds", UNSET)
        for componentsschemasv0_0_43_shares_uint64_tres_list_item_data in _run_seconds or []:
            componentsschemasv0_0_43_shares_uint64_tres_list_item = V0043SharesUint64Tres.from_dict(
                componentsschemasv0_0_43_shares_uint64_tres_list_item_data
            )

            run_seconds.append(componentsschemasv0_0_43_shares_uint64_tres_list_item)

        group_minutes = []
        _group_minutes = d.pop("group_minutes", UNSET)
        for componentsschemasv0_0_43_shares_uint64_tres_list_item_data in _group_minutes or []:
            componentsschemasv0_0_43_shares_uint64_tres_list_item = V0043SharesUint64Tres.from_dict(
                componentsschemasv0_0_43_shares_uint64_tres_list_item_data
            )

            group_minutes.append(componentsschemasv0_0_43_shares_uint64_tres_list_item)

        usage = []
        _usage = d.pop("usage", UNSET)
        for componentsschemasv0_0_43_shares_float128_tres_list_item_data in _usage or []:
            componentsschemasv0_0_43_shares_float128_tres_list_item = V0043SharesFloat128Tres.from_dict(
                componentsschemasv0_0_43_shares_float128_tres_list_item_data
            )

            usage.append(componentsschemasv0_0_43_shares_float128_tres_list_item)

        v0043_assoc_shares_obj_wrap_tres = cls(
            run_seconds=run_seconds,
            group_minutes=group_minutes,
            usage=usage,
        )

        v0043_assoc_shares_obj_wrap_tres.additional_properties = d
        return v0043_assoc_shares_obj_wrap_tres

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
