from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_qos_limits_max_accruing import V0043QosLimitsMaxAccruing
    from ..models.v0043_qos_limits_max_active_jobs import V0043QosLimitsMaxActiveJobs
    from ..models.v0043_qos_limits_max_jobs import V0043QosLimitsMaxJobs
    from ..models.v0043_qos_limits_max_tres import V0043QosLimitsMaxTres
    from ..models.v0043_qos_limits_max_wall_clock import V0043QosLimitsMaxWallClock


T = TypeVar("T", bound="V0043QosLimitsMax")


@_attrs_define
class V0043QosLimitsMax:
    """
    Attributes:
        active_jobs (Union[Unset, V0043QosLimitsMaxActiveJobs]):
        jobs (Union[Unset, V0043QosLimitsMaxJobs]):
        tres (Union[Unset, V0043QosLimitsMaxTres]):
        wall_clock (Union[Unset, V0043QosLimitsMaxWallClock]):
        accruing (Union[Unset, V0043QosLimitsMaxAccruing]):
    """

    active_jobs: Union[Unset, "V0043QosLimitsMaxActiveJobs"] = UNSET
    jobs: Union[Unset, "V0043QosLimitsMaxJobs"] = UNSET
    tres: Union[Unset, "V0043QosLimitsMaxTres"] = UNSET
    wall_clock: Union[Unset, "V0043QosLimitsMaxWallClock"] = UNSET
    accruing: Union[Unset, "V0043QosLimitsMaxAccruing"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        active_jobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.active_jobs, Unset):
            active_jobs = self.active_jobs.to_dict()

        jobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.jobs, Unset):
            jobs = self.jobs.to_dict()

        tres: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = self.tres.to_dict()

        wall_clock: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.wall_clock, Unset):
            wall_clock = self.wall_clock.to_dict()

        accruing: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.accruing, Unset):
            accruing = self.accruing.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if active_jobs is not UNSET:
            field_dict["active_jobs"] = active_jobs
        if jobs is not UNSET:
            field_dict["jobs"] = jobs
        if tres is not UNSET:
            field_dict["tres"] = tres
        if wall_clock is not UNSET:
            field_dict["wall_clock"] = wall_clock
        if accruing is not UNSET:
            field_dict["accruing"] = accruing

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_qos_limits_max_accruing import V0043QosLimitsMaxAccruing
        from ..models.v0043_qos_limits_max_active_jobs import V0043QosLimitsMaxActiveJobs
        from ..models.v0043_qos_limits_max_jobs import V0043QosLimitsMaxJobs
        from ..models.v0043_qos_limits_max_tres import V0043QosLimitsMaxTres
        from ..models.v0043_qos_limits_max_wall_clock import V0043QosLimitsMaxWallClock

        d = dict(src_dict)
        _active_jobs = d.pop("active_jobs", UNSET)
        active_jobs: Union[Unset, V0043QosLimitsMaxActiveJobs]
        if isinstance(_active_jobs, Unset):
            active_jobs = UNSET
        else:
            active_jobs = V0043QosLimitsMaxActiveJobs.from_dict(_active_jobs)

        _jobs = d.pop("jobs", UNSET)
        jobs: Union[Unset, V0043QosLimitsMaxJobs]
        if isinstance(_jobs, Unset):
            jobs = UNSET
        else:
            jobs = V0043QosLimitsMaxJobs.from_dict(_jobs)

        _tres = d.pop("tres", UNSET)
        tres: Union[Unset, V0043QosLimitsMaxTres]
        if isinstance(_tres, Unset):
            tres = UNSET
        else:
            tres = V0043QosLimitsMaxTres.from_dict(_tres)

        _wall_clock = d.pop("wall_clock", UNSET)
        wall_clock: Union[Unset, V0043QosLimitsMaxWallClock]
        if isinstance(_wall_clock, Unset):
            wall_clock = UNSET
        else:
            wall_clock = V0043QosLimitsMaxWallClock.from_dict(_wall_clock)

        _accruing = d.pop("accruing", UNSET)
        accruing: Union[Unset, V0043QosLimitsMaxAccruing]
        if isinstance(_accruing, Unset):
            accruing = UNSET
        else:
            accruing = V0043QosLimitsMaxAccruing.from_dict(_accruing)

        v0043_qos_limits_max = cls(
            active_jobs=active_jobs,
            jobs=jobs,
            tres=tres,
            wall_clock=wall_clock,
            accruing=accruing,
        )

        v0043_qos_limits_max.additional_properties = d
        return v0043_qos_limits_max

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
