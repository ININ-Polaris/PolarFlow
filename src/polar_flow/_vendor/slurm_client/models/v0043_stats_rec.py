from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_rollup_stats import V0043RollupStats
    from ..models.v0043_stats_rpc import V0043StatsRpc
    from ..models.v0043_stats_user import V0043StatsUser


T = TypeVar("T", bound="V0043StatsRec")


@_attrs_define
class V0043StatsRec:
    """
    Attributes:
        time_start (Union[Unset, int]): When data collection started (UNIX timestamp) (UNIX timestamp or time string
            recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
        rollups (Union[Unset, V0043RollupStats]):
        rp_cs (Union[Unset, list['V0043StatsRpc']]):
        users (Union[Unset, list['V0043StatsUser']]):
    """

    time_start: Union[Unset, int] = UNSET
    rollups: Union[Unset, "V0043RollupStats"] = UNSET
    rp_cs: Union[Unset, list["V0043StatsRpc"]] = UNSET
    users: Union[Unset, list["V0043StatsUser"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        time_start = self.time_start

        rollups: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.rollups, Unset):
            rollups = self.rollups.to_dict()

        rp_cs: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.rp_cs, Unset):
            rp_cs = []
            for componentsschemasv0_0_43_stats_rpc_list_item_data in self.rp_cs:
                componentsschemasv0_0_43_stats_rpc_list_item = (
                    componentsschemasv0_0_43_stats_rpc_list_item_data.to_dict()
                )
                rp_cs.append(componentsschemasv0_0_43_stats_rpc_list_item)

        users: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.users, Unset):
            users = []
            for componentsschemasv0_0_43_stats_user_list_item_data in self.users:
                componentsschemasv0_0_43_stats_user_list_item = (
                    componentsschemasv0_0_43_stats_user_list_item_data.to_dict()
                )
                users.append(componentsschemasv0_0_43_stats_user_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if time_start is not UNSET:
            field_dict["time_start"] = time_start
        if rollups is not UNSET:
            field_dict["rollups"] = rollups
        if rp_cs is not UNSET:
            field_dict["RPCs"] = rp_cs
        if users is not UNSET:
            field_dict["users"] = users

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_rollup_stats import V0043RollupStats
        from ..models.v0043_stats_rpc import V0043StatsRpc
        from ..models.v0043_stats_user import V0043StatsUser

        d = dict(src_dict)
        time_start = d.pop("time_start", UNSET)

        _rollups = d.pop("rollups", UNSET)
        rollups: Union[Unset, V0043RollupStats]
        if isinstance(_rollups, Unset):
            rollups = UNSET
        else:
            rollups = V0043RollupStats.from_dict(_rollups)

        rp_cs = []
        _rp_cs = d.pop("RPCs", UNSET)
        for componentsschemasv0_0_43_stats_rpc_list_item_data in _rp_cs or []:
            componentsschemasv0_0_43_stats_rpc_list_item = V0043StatsRpc.from_dict(
                componentsschemasv0_0_43_stats_rpc_list_item_data
            )

            rp_cs.append(componentsschemasv0_0_43_stats_rpc_list_item)

        users = []
        _users = d.pop("users", UNSET)
        for componentsschemasv0_0_43_stats_user_list_item_data in _users or []:
            componentsschemasv0_0_43_stats_user_list_item = V0043StatsUser.from_dict(
                componentsschemasv0_0_43_stats_user_list_item_data
            )

            users.append(componentsschemasv0_0_43_stats_user_list_item)

        v0043_stats_rec = cls(
            time_start=time_start,
            rollups=rollups,
            rp_cs=rp_cs,
            users=users,
        )

        v0043_stats_rec.additional_properties = d
        return v0043_stats_rec

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
