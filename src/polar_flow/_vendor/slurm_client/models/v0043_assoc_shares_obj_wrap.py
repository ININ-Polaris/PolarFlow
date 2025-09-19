from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_assoc_shares_obj_wrap_type_item import V0043AssocSharesObjWrapTypeItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_assoc_shares_obj_wrap_fairshare import V0043AssocSharesObjWrapFairshare
    from ..models.v0043_assoc_shares_obj_wrap_tres import V0043AssocSharesObjWrapTres
    from ..models.v0043_float_64_no_val_struct import V0043Float64NoValStruct
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043AssocSharesObjWrap")


@_attrs_define
class V0043AssocSharesObjWrap:
    """
    Attributes:
        id (Union[Unset, int]): Association ID
        cluster (Union[Unset, str]): Cluster name
        name (Union[Unset, str]): Share name
        parent (Union[Unset, str]): Parent name
        partition (Union[Unset, str]): Partition name
        shares_normalized (Union[Unset, V0043Float64NoValStruct]):
        shares (Union[Unset, V0043Uint32NoValStruct]):
        tres (Union[Unset, V0043AssocSharesObjWrapTres]):
        effective_usage (Union[Unset, V0043Float64NoValStruct]):
        usage_normalized (Union[Unset, V0043Float64NoValStruct]):
        usage (Union[Unset, int]): Measure of tresbillableunits usage
        fairshare (Union[Unset, V0043AssocSharesObjWrapFairshare]):
        type_ (Union[Unset, list[V0043AssocSharesObjWrapTypeItem]]): User or account association
    """

    id: Union[Unset, int] = UNSET
    cluster: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    parent: Union[Unset, str] = UNSET
    partition: Union[Unset, str] = UNSET
    shares_normalized: Union[Unset, "V0043Float64NoValStruct"] = UNSET
    shares: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    tres: Union[Unset, "V0043AssocSharesObjWrapTres"] = UNSET
    effective_usage: Union[Unset, "V0043Float64NoValStruct"] = UNSET
    usage_normalized: Union[Unset, "V0043Float64NoValStruct"] = UNSET
    usage: Union[Unset, int] = UNSET
    fairshare: Union[Unset, "V0043AssocSharesObjWrapFairshare"] = UNSET
    type_: Union[Unset, list[V0043AssocSharesObjWrapTypeItem]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        cluster = self.cluster

        name = self.name

        parent = self.parent

        partition = self.partition

        shares_normalized: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.shares_normalized, Unset):
            shares_normalized = self.shares_normalized.to_dict()

        shares: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.shares, Unset):
            shares = self.shares.to_dict()

        tres: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = self.tres.to_dict()

        effective_usage: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.effective_usage, Unset):
            effective_usage = self.effective_usage.to_dict()

        usage_normalized: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.usage_normalized, Unset):
            usage_normalized = self.usage_normalized.to_dict()

        usage = self.usage

        fairshare: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.fairshare, Unset):
            fairshare = self.fairshare.to_dict()

        type_: Union[Unset, list[str]] = UNSET
        if not isinstance(self.type_, Unset):
            type_ = []
            for type_item_data in self.type_:
                type_item = type_item_data.value
                type_.append(type_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if cluster is not UNSET:
            field_dict["cluster"] = cluster
        if name is not UNSET:
            field_dict["name"] = name
        if parent is not UNSET:
            field_dict["parent"] = parent
        if partition is not UNSET:
            field_dict["partition"] = partition
        if shares_normalized is not UNSET:
            field_dict["shares_normalized"] = shares_normalized
        if shares is not UNSET:
            field_dict["shares"] = shares
        if tres is not UNSET:
            field_dict["tres"] = tres
        if effective_usage is not UNSET:
            field_dict["effective_usage"] = effective_usage
        if usage_normalized is not UNSET:
            field_dict["usage_normalized"] = usage_normalized
        if usage is not UNSET:
            field_dict["usage"] = usage
        if fairshare is not UNSET:
            field_dict["fairshare"] = fairshare
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_assoc_shares_obj_wrap_fairshare import V0043AssocSharesObjWrapFairshare
        from ..models.v0043_assoc_shares_obj_wrap_tres import V0043AssocSharesObjWrapTres
        from ..models.v0043_float_64_no_val_struct import V0043Float64NoValStruct
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        id = d.pop("id", UNSET)

        cluster = d.pop("cluster", UNSET)

        name = d.pop("name", UNSET)

        parent = d.pop("parent", UNSET)

        partition = d.pop("partition", UNSET)

        _shares_normalized = d.pop("shares_normalized", UNSET)
        shares_normalized: Union[Unset, V0043Float64NoValStruct]
        if isinstance(_shares_normalized, Unset):
            shares_normalized = UNSET
        else:
            shares_normalized = V0043Float64NoValStruct.from_dict(_shares_normalized)

        _shares = d.pop("shares", UNSET)
        shares: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_shares, Unset):
            shares = UNSET
        else:
            shares = V0043Uint32NoValStruct.from_dict(_shares)

        _tres = d.pop("tres", UNSET)
        tres: Union[Unset, V0043AssocSharesObjWrapTres]
        if isinstance(_tres, Unset):
            tres = UNSET
        else:
            tres = V0043AssocSharesObjWrapTres.from_dict(_tres)

        _effective_usage = d.pop("effective_usage", UNSET)
        effective_usage: Union[Unset, V0043Float64NoValStruct]
        if isinstance(_effective_usage, Unset):
            effective_usage = UNSET
        else:
            effective_usage = V0043Float64NoValStruct.from_dict(_effective_usage)

        _usage_normalized = d.pop("usage_normalized", UNSET)
        usage_normalized: Union[Unset, V0043Float64NoValStruct]
        if isinstance(_usage_normalized, Unset):
            usage_normalized = UNSET
        else:
            usage_normalized = V0043Float64NoValStruct.from_dict(_usage_normalized)

        usage = d.pop("usage", UNSET)

        _fairshare = d.pop("fairshare", UNSET)
        fairshare: Union[Unset, V0043AssocSharesObjWrapFairshare]
        if isinstance(_fairshare, Unset):
            fairshare = UNSET
        else:
            fairshare = V0043AssocSharesObjWrapFairshare.from_dict(_fairshare)

        type_ = []
        _type_ = d.pop("type", UNSET)
        for type_item_data in _type_ or []:
            type_item = V0043AssocSharesObjWrapTypeItem(type_item_data)

            type_.append(type_item)

        v0043_assoc_shares_obj_wrap = cls(
            id=id,
            cluster=cluster,
            name=name,
            parent=parent,
            partition=partition,
            shares_normalized=shares_normalized,
            shares=shares,
            tres=tres,
            effective_usage=effective_usage,
            usage_normalized=usage_normalized,
            usage=usage,
            fairshare=fairshare,
            type_=type_,
        )

        v0043_assoc_shares_obj_wrap.additional_properties = d
        return v0043_assoc_shares_obj_wrap

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
