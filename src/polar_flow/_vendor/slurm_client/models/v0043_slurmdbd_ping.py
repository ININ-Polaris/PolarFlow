from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="V0043SlurmdbdPing")


@_attrs_define
class V0043SlurmdbdPing:
    hostname: str
    """ Target for ping """
    responding: bool
    """ If ping RPC responded with pong from slurmdbd """
    latency: int
    """ Number of microseconds it took to successfully ping or timeout """
    primary: bool
    """ Is responding slurmdbd the primary controller (Is responding slurmctld the primary controller) """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        hostname = self.hostname

        responding = self.responding

        latency = self.latency

        primary = self.primary

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "hostname": hostname,
                "responding": responding,
                "latency": latency,
                "primary": primary,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        hostname = d.pop("hostname")

        responding = d.pop("responding")

        latency = d.pop("latency")

        primary = d.pop("primary")

        v0043_slurmdbd_ping = cls(
            hostname=hostname,
            responding=responding,
            latency=latency,
            primary=primary,
        )

        v0043_slurmdbd_ping.additional_properties = d
        return v0043_slurmdbd_ping

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
