from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043StepNodes")


@_attrs_define
class V0043StepNodes:
    """
    Attributes:
        count (Union[Unset, int]): Number of nodes in the job step
        range_ (Union[Unset, str]): Node(s) allocated to the job step
        list_ (Union[Unset, list[str]]):
    """

    count: Union[Unset, int] = UNSET
    range_: Union[Unset, str] = UNSET
    list_: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        count = self.count

        range_ = self.range_

        list_: Union[Unset, list[str]] = UNSET
        if not isinstance(self.list_, Unset):
            list_ = self.list_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if count is not UNSET:
            field_dict["count"] = count
        if range_ is not UNSET:
            field_dict["range"] = range_
        if list_ is not UNSET:
            field_dict["list"] = list_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        count = d.pop("count", UNSET)

        range_ = d.pop("range", UNSET)

        list_ = cast(list[str], d.pop("list", UNSET))

        v0043_step_nodes = cls(
            count=count,
            range_=range_,
            list_=list_,
        )

        v0043_step_nodes.additional_properties = d
        return v0043_step_nodes

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
