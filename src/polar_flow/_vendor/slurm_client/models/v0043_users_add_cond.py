from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_assoc_rec_set import V0043AssocRecSet


T = TypeVar("T", bound="V0043UsersAddCond")


@_attrs_define
class V0043UsersAddCond:
    """
    Attributes:
        users (list[str]):
        accounts (Union[Unset, list[str]]):
        association (Union[Unset, V0043AssocRecSet]):
        clusters (Union[Unset, list[str]]):
        partitions (Union[Unset, list[str]]):
        wckeys (Union[Unset, list[str]]):
    """

    users: list[str]
    accounts: Union[Unset, list[str]] = UNSET
    association: Union[Unset, "V0043AssocRecSet"] = UNSET
    clusters: Union[Unset, list[str]] = UNSET
    partitions: Union[Unset, list[str]] = UNSET
    wckeys: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        users = self.users

        accounts: Union[Unset, list[str]] = UNSET
        if not isinstance(self.accounts, Unset):
            accounts = self.accounts

        association: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.association, Unset):
            association = self.association.to_dict()

        clusters: Union[Unset, list[str]] = UNSET
        if not isinstance(self.clusters, Unset):
            clusters = self.clusters

        partitions: Union[Unset, list[str]] = UNSET
        if not isinstance(self.partitions, Unset):
            partitions = self.partitions

        wckeys: Union[Unset, list[str]] = UNSET
        if not isinstance(self.wckeys, Unset):
            wckeys = self.wckeys

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "users": users,
            }
        )
        if accounts is not UNSET:
            field_dict["accounts"] = accounts
        if association is not UNSET:
            field_dict["association"] = association
        if clusters is not UNSET:
            field_dict["clusters"] = clusters
        if partitions is not UNSET:
            field_dict["partitions"] = partitions
        if wckeys is not UNSET:
            field_dict["wckeys"] = wckeys

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_assoc_rec_set import V0043AssocRecSet

        d = dict(src_dict)
        users = cast(list[str], d.pop("users"))

        accounts = cast(list[str], d.pop("accounts", UNSET))

        _association = d.pop("association", UNSET)
        association: Union[Unset, V0043AssocRecSet]
        if isinstance(_association, Unset):
            association = UNSET
        else:
            association = V0043AssocRecSet.from_dict(_association)

        clusters = cast(list[str], d.pop("clusters", UNSET))

        partitions = cast(list[str], d.pop("partitions", UNSET))

        wckeys = cast(list[str], d.pop("wckeys", UNSET))

        v0043_users_add_cond = cls(
            users=users,
            accounts=accounts,
            association=association,
            clusters=clusters,
            partitions=partitions,
            wckeys=wckeys,
        )

        v0043_users_add_cond.additional_properties = d
        return v0043_users_add_cond

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
