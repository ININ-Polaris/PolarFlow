from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043AcctGatherEnergy")


@_attrs_define
class V0043AcctGatherEnergy:
    """
    Attributes:
        average_watts (Union[Unset, int]): Average power consumption, in watts
        base_consumed_energy (Union[Unset, int]): The energy consumed between when the node was powered on and the last
            time it was registered by slurmd, in joules
        consumed_energy (Union[Unset, int]): The energy consumed between the last time the node was registered by the
            slurmd daemon and the last node energy accounting sample, in joules
        current_watts (Union[Unset, V0043Uint32NoValStruct]):
        previous_consumed_energy (Union[Unset, int]): Previous value of consumed_energy
        last_collected (Union[Unset, int]): Time when energy data was last retrieved (UNIX timestamp) (UNIX timestamp or
            time string recognized by Slurm (e.g., '[MM/DD[/YY]-]HH:MM[:SS]'))
    """

    average_watts: Union[Unset, int] = UNSET
    base_consumed_energy: Union[Unset, int] = UNSET
    consumed_energy: Union[Unset, int] = UNSET
    current_watts: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    previous_consumed_energy: Union[Unset, int] = UNSET
    last_collected: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        average_watts = self.average_watts

        base_consumed_energy = self.base_consumed_energy

        consumed_energy = self.consumed_energy

        current_watts: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.current_watts, Unset):
            current_watts = self.current_watts.to_dict()

        previous_consumed_energy = self.previous_consumed_energy

        last_collected = self.last_collected

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if average_watts is not UNSET:
            field_dict["average_watts"] = average_watts
        if base_consumed_energy is not UNSET:
            field_dict["base_consumed_energy"] = base_consumed_energy
        if consumed_energy is not UNSET:
            field_dict["consumed_energy"] = consumed_energy
        if current_watts is not UNSET:
            field_dict["current_watts"] = current_watts
        if previous_consumed_energy is not UNSET:
            field_dict["previous_consumed_energy"] = previous_consumed_energy
        if last_collected is not UNSET:
            field_dict["last_collected"] = last_collected

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        average_watts = d.pop("average_watts", UNSET)

        base_consumed_energy = d.pop("base_consumed_energy", UNSET)

        consumed_energy = d.pop("consumed_energy", UNSET)

        _current_watts = d.pop("current_watts", UNSET)
        current_watts: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_current_watts, Unset):
            current_watts = UNSET
        else:
            current_watts = V0043Uint32NoValStruct.from_dict(_current_watts)

        previous_consumed_energy = d.pop("previous_consumed_energy", UNSET)

        last_collected = d.pop("last_collected", UNSET)

        v0043_acct_gather_energy = cls(
            average_watts=average_watts,
            base_consumed_energy=base_consumed_energy,
            consumed_energy=consumed_energy,
            current_watts=current_watts,
            previous_consumed_energy=previous_consumed_energy,
            last_collected=last_collected,
        )

        v0043_acct_gather_energy.additional_properties = d
        return v0043_acct_gather_energy

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
