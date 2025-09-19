from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_assoc_short import V0043AssocShort


T = TypeVar("T", bound="V0043ClusterRecAssociations")


@_attrs_define
class V0043ClusterRecAssociations:
    """
    Attributes:
        root (Union[Unset, V0043AssocShort]):
    """

    root: Union[Unset, "V0043AssocShort"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        root: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.root, Unset):
            root = self.root.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if root is not UNSET:
            field_dict["root"] = root

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_assoc_short import V0043AssocShort

        d = dict(src_dict)
        _root = d.pop("root", UNSET)
        root: Union[Unset, V0043AssocShort]
        if isinstance(_root, Unset):
            root = UNSET
        else:
            root = V0043AssocShort.from_dict(_root)

        v0043_cluster_rec_associations = cls(
            root=root,
        )

        v0043_cluster_rec_associations.additional_properties = d
        return v0043_cluster_rec_associations

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
