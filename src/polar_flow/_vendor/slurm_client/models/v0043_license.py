from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043License")


@_attrs_define
class V0043License:
    """
    Attributes:
        license_name (Union[Unset, str]): Name of the license
        total (Union[Unset, int]): Total number of licenses present
        used (Union[Unset, int]): Number of licenses in use
        free (Union[Unset, int]): Number of licenses currently available
        remote (Union[Unset, bool]): Indicates whether licenses are served by the database
        reserved (Union[Unset, int]): Number of licenses reserved
        last_consumed (Union[Unset, int]): Last known number of licenses that were consumed in the license manager
            (Remote Only)
        last_deficit (Union[Unset, int]): Number of "missing licenses" from the cluster's perspective
        last_update (Union[Unset, int]): When the license information was last updated (UNIX Timestamp) (UNIX timestamp
            or time string recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
    """

    license_name: Union[Unset, str] = UNSET
    total: Union[Unset, int] = UNSET
    used: Union[Unset, int] = UNSET
    free: Union[Unset, int] = UNSET
    remote: Union[Unset, bool] = UNSET
    reserved: Union[Unset, int] = UNSET
    last_consumed: Union[Unset, int] = UNSET
    last_deficit: Union[Unset, int] = UNSET
    last_update: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        license_name = self.license_name

        total = self.total

        used = self.used

        free = self.free

        remote = self.remote

        reserved = self.reserved

        last_consumed = self.last_consumed

        last_deficit = self.last_deficit

        last_update = self.last_update

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if license_name is not UNSET:
            field_dict["LicenseName"] = license_name
        if total is not UNSET:
            field_dict["Total"] = total
        if used is not UNSET:
            field_dict["Used"] = used
        if free is not UNSET:
            field_dict["Free"] = free
        if remote is not UNSET:
            field_dict["Remote"] = remote
        if reserved is not UNSET:
            field_dict["Reserved"] = reserved
        if last_consumed is not UNSET:
            field_dict["LastConsumed"] = last_consumed
        if last_deficit is not UNSET:
            field_dict["LastDeficit"] = last_deficit
        if last_update is not UNSET:
            field_dict["LastUpdate"] = last_update

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        license_name = d.pop("LicenseName", UNSET)

        total = d.pop("Total", UNSET)

        used = d.pop("Used", UNSET)

        free = d.pop("Free", UNSET)

        remote = d.pop("Remote", UNSET)

        reserved = d.pop("Reserved", UNSET)

        last_consumed = d.pop("LastConsumed", UNSET)

        last_deficit = d.pop("LastDeficit", UNSET)

        last_update = d.pop("LastUpdate", UNSET)

        v0043_license = cls(
            license_name=license_name,
            total=total,
            used=used,
            free=free,
            remote=remote,
            reserved=reserved,
            last_consumed=last_consumed,
            last_deficit=last_deficit,
            last_update=last_update,
        )

        v0043_license.additional_properties = d
        return v0043_license

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
