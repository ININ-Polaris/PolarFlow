from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_assoc_flags_item import V0043AssocFlagsItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_accounting import V0043Accounting
    from ..models.v0043_assoc_default import V0043AssocDefault
    from ..models.v0043_assoc_max import V0043AssocMax
    from ..models.v0043_assoc_min import V0043AssocMin
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043Assoc")


@_attrs_define
class V0043Assoc:
    """
    Attributes:
        user (str): User name
        accounting (Union[Unset, list['V0043Accounting']]):
        account (Union[Unset, str]): Account name
        cluster (Union[Unset, str]): Cluster name
        comment (Union[Unset, str]): Arbitrary comment
        default (Union[Unset, V0043AssocDefault]):
        flags (Union[Unset, list[V0043AssocFlagsItem]]): Flags on the association
        max_ (Union[Unset, V0043AssocMax]):
        id (Union[Unset, int]): Unique ID (Association ID)
        is_default (Union[Unset, bool]): Is default association for user
        lineage (Union[Unset, str]): Complete path up the hierarchy to the root association
        min_ (Union[Unset, V0043AssocMin]):
        parent_account (Union[Unset, str]): Name of parent account
        partition (Union[Unset, str]): Partition name
        priority (Union[Unset, V0043Uint32NoValStruct]):
        qos (Union[Unset, list[str]]): List of QOS names
        shares_raw (Union[Unset, int]): Allocated shares used for fairshare calculation
    """

    user: str
    accounting: Union[Unset, list["V0043Accounting"]] = UNSET
    account: Union[Unset, str] = UNSET
    cluster: Union[Unset, str] = UNSET
    comment: Union[Unset, str] = UNSET
    default: Union[Unset, "V0043AssocDefault"] = UNSET
    flags: Union[Unset, list[V0043AssocFlagsItem]] = UNSET
    max_: Union[Unset, "V0043AssocMax"] = UNSET
    id: Union[Unset, int] = UNSET
    is_default: Union[Unset, bool] = UNSET
    lineage: Union[Unset, str] = UNSET
    min_: Union[Unset, "V0043AssocMin"] = UNSET
    parent_account: Union[Unset, str] = UNSET
    partition: Union[Unset, str] = UNSET
    priority: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    qos: Union[Unset, list[str]] = UNSET
    shares_raw: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user = self.user

        accounting: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.accounting, Unset):
            accounting = []
            for componentsschemasv0_0_43_accounting_list_item_data in self.accounting:
                componentsschemasv0_0_43_accounting_list_item = (
                    componentsschemasv0_0_43_accounting_list_item_data.to_dict()
                )
                accounting.append(componentsschemasv0_0_43_accounting_list_item)

        account = self.account

        cluster = self.cluster

        comment = self.comment

        default: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.default, Unset):
            default = self.default.to_dict()

        flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.flags, Unset):
            flags = []
            for flags_item_data in self.flags:
                flags_item = flags_item_data.value
                flags.append(flags_item)

        max_: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.max_, Unset):
            max_ = self.max_.to_dict()

        id = self.id

        is_default = self.is_default

        lineage = self.lineage

        min_: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.min_, Unset):
            min_ = self.min_.to_dict()

        parent_account = self.parent_account

        partition = self.partition

        priority: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.priority, Unset):
            priority = self.priority.to_dict()

        qos: Union[Unset, list[str]] = UNSET
        if not isinstance(self.qos, Unset):
            qos = self.qos

        shares_raw = self.shares_raw

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user": user,
            }
        )
        if accounting is not UNSET:
            field_dict["accounting"] = accounting
        if account is not UNSET:
            field_dict["account"] = account
        if cluster is not UNSET:
            field_dict["cluster"] = cluster
        if comment is not UNSET:
            field_dict["comment"] = comment
        if default is not UNSET:
            field_dict["default"] = default
        if flags is not UNSET:
            field_dict["flags"] = flags
        if max_ is not UNSET:
            field_dict["max"] = max_
        if id is not UNSET:
            field_dict["id"] = id
        if is_default is not UNSET:
            field_dict["is_default"] = is_default
        if lineage is not UNSET:
            field_dict["lineage"] = lineage
        if min_ is not UNSET:
            field_dict["min"] = min_
        if parent_account is not UNSET:
            field_dict["parent_account"] = parent_account
        if partition is not UNSET:
            field_dict["partition"] = partition
        if priority is not UNSET:
            field_dict["priority"] = priority
        if qos is not UNSET:
            field_dict["qos"] = qos
        if shares_raw is not UNSET:
            field_dict["shares_raw"] = shares_raw

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_accounting import V0043Accounting
        from ..models.v0043_assoc_default import V0043AssocDefault
        from ..models.v0043_assoc_max import V0043AssocMax
        from ..models.v0043_assoc_min import V0043AssocMin
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        user = d.pop("user")

        accounting = []
        _accounting = d.pop("accounting", UNSET)
        for componentsschemasv0_0_43_accounting_list_item_data in _accounting or []:
            componentsschemasv0_0_43_accounting_list_item = V0043Accounting.from_dict(
                componentsschemasv0_0_43_accounting_list_item_data
            )

            accounting.append(componentsschemasv0_0_43_accounting_list_item)

        account = d.pop("account", UNSET)

        cluster = d.pop("cluster", UNSET)

        comment = d.pop("comment", UNSET)

        _default = d.pop("default", UNSET)
        default: Union[Unset, V0043AssocDefault]
        if isinstance(_default, Unset):
            default = UNSET
        else:
            default = V0043AssocDefault.from_dict(_default)

        flags = []
        _flags = d.pop("flags", UNSET)
        for flags_item_data in _flags or []:
            flags_item = V0043AssocFlagsItem(flags_item_data)

            flags.append(flags_item)

        _max_ = d.pop("max", UNSET)
        max_: Union[Unset, V0043AssocMax]
        if isinstance(_max_, Unset):
            max_ = UNSET
        else:
            max_ = V0043AssocMax.from_dict(_max_)

        id = d.pop("id", UNSET)

        is_default = d.pop("is_default", UNSET)

        lineage = d.pop("lineage", UNSET)

        _min_ = d.pop("min", UNSET)
        min_: Union[Unset, V0043AssocMin]
        if isinstance(_min_, Unset):
            min_ = UNSET
        else:
            min_ = V0043AssocMin.from_dict(_min_)

        parent_account = d.pop("parent_account", UNSET)

        partition = d.pop("partition", UNSET)

        _priority = d.pop("priority", UNSET)
        priority: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_priority, Unset):
            priority = UNSET
        else:
            priority = V0043Uint32NoValStruct.from_dict(_priority)

        qos = cast(list[str], d.pop("qos", UNSET))

        shares_raw = d.pop("shares_raw", UNSET)

        v0043_assoc = cls(
            user=user,
            accounting=accounting,
            account=account,
            cluster=cluster,
            comment=comment,
            default=default,
            flags=flags,
            max_=max_,
            id=id,
            is_default=is_default,
            lineage=lineage,
            min_=min_,
            parent_account=parent_account,
            partition=partition,
            priority=priority,
            qos=qos,
            shares_raw=shares_raw,
        )

        v0043_assoc.additional_properties = d
        return v0043_assoc

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
