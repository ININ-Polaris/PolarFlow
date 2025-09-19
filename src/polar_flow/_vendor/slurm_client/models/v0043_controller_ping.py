from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043ControllerPing")


@_attrs_define
class V0043ControllerPing:
    """
    Attributes:
        responding (bool): If ping RPC responded with pong from controller
        primary (bool): Is responding slurmctld the primary controller (Is responding slurmctld the primary controller)
        hostname (Union[Unset, str]): Target for ping
        pinged (Union[Unset, str]): Ping result
        latency (Union[Unset, int]): Number of microseconds it took to successfully ping or timeout
        mode (Union[Unset, str]): The operating mode of the responding slurmctld
    """

    responding: bool
    primary: bool
    hostname: Union[Unset, str] = UNSET
    pinged: Union[Unset, str] = UNSET
    latency: Union[Unset, int] = UNSET
    mode: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        responding = self.responding

        primary = self.primary

        hostname = self.hostname

        pinged = self.pinged

        latency = self.latency

        mode = self.mode

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "responding": responding,
                "primary": primary,
            }
        )
        if hostname is not UNSET:
            field_dict["hostname"] = hostname
        if pinged is not UNSET:
            field_dict["pinged"] = pinged
        if latency is not UNSET:
            field_dict["latency"] = latency
        if mode is not UNSET:
            field_dict["mode"] = mode

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        responding = d.pop("responding")

        primary = d.pop("primary")

        hostname = d.pop("hostname", UNSET)

        pinged = d.pop("pinged", UNSET)

        latency = d.pop("latency", UNSET)

        mode = d.pop("mode", UNSET)

        v0043_controller_ping = cls(
            responding=responding,
            primary=primary,
            hostname=hostname,
            pinged=pinged,
            latency=latency,
            mode=mode,
        )

        v0043_controller_ping.additional_properties = d
        return v0043_controller_ping

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
