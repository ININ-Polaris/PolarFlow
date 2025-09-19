from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_job_time_system import V0043JobTimeSystem
    from ..models.v0043_job_time_total import V0043JobTimeTotal
    from ..models.v0043_job_time_user import V0043JobTimeUser
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
    from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct


T = TypeVar("T", bound="V0043JobTime")


@_attrs_define
class V0043JobTime:
    elapsed: Union[Unset, int] = UNSET
    """ Elapsed time in seconds """
    eligible: Union[Unset, int] = UNSET
    """ Time when the job became eligible to run (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm
    (e.g., '[MM/DD[/YY]-]HH:MM[:SS]')) """
    end: Union[Unset, int] = UNSET
    """ End time (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g.,
    '[MM/DD[/YY]-]HH:MM[:SS]')) """
    planned: Union[Unset, "V0043Uint64NoValStruct"] = UNSET
    start: Union[Unset, int] = UNSET
    """ Time execution began (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g.,
    '[MM/DD[/YY]-]HH:MM[:SS]')) """
    submission: Union[Unset, int] = UNSET
    """ Time when the job was submitted (UNIX timestamp) (UNIX timestamp or time string recognized by Slurm (e.g.,
    '[MM/DD[/YY]-]HH:MM[:SS]')) """
    suspended: Union[Unset, int] = UNSET
    """ Total time in suspended state in seconds """
    system: Union[Unset, "V0043JobTimeSystem"] = UNSET
    limit: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    total: Union[Unset, "V0043JobTimeTotal"] = UNSET
    user: Union[Unset, "V0043JobTimeUser"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        elapsed = self.elapsed

        eligible = self.eligible

        end = self.end

        planned: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.planned, Unset):
            planned = self.planned.to_dict()

        start = self.start

        submission = self.submission

        suspended = self.suspended

        system: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.system, Unset):
            system = self.system.to_dict()

        limit: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.limit, Unset):
            limit = self.limit.to_dict()

        total: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.total, Unset):
            total = self.total.to_dict()

        user: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.user, Unset):
            user = self.user.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if elapsed is not UNSET:
            field_dict["elapsed"] = elapsed
        if eligible is not UNSET:
            field_dict["eligible"] = eligible
        if end is not UNSET:
            field_dict["end"] = end
        if planned is not UNSET:
            field_dict["planned"] = planned
        if start is not UNSET:
            field_dict["start"] = start
        if submission is not UNSET:
            field_dict["submission"] = submission
        if suspended is not UNSET:
            field_dict["suspended"] = suspended
        if system is not UNSET:
            field_dict["system"] = system
        if limit is not UNSET:
            field_dict["limit"] = limit
        if total is not UNSET:
            field_dict["total"] = total
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_job_time_system import V0043JobTimeSystem
        from ..models.v0043_job_time_total import V0043JobTimeTotal
        from ..models.v0043_job_time_user import V0043JobTimeUser
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct
        from ..models.v0043_uint_64_no_val_struct import V0043Uint64NoValStruct

        d = dict(src_dict)
        elapsed = d.pop("elapsed", UNSET)

        eligible = d.pop("eligible", UNSET)

        end = d.pop("end", UNSET)

        _planned = d.pop("planned", UNSET)
        planned: Union[Unset, V0043Uint64NoValStruct]
        if isinstance(_planned, Unset):
            planned = UNSET
        else:
            planned = V0043Uint64NoValStruct.from_dict(_planned)

        start = d.pop("start", UNSET)

        submission = d.pop("submission", UNSET)

        suspended = d.pop("suspended", UNSET)

        _system = d.pop("system", UNSET)
        system: Union[Unset, V0043JobTimeSystem]
        if isinstance(_system, Unset):
            system = UNSET
        else:
            system = V0043JobTimeSystem.from_dict(_system)

        _limit = d.pop("limit", UNSET)
        limit: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_limit, Unset):
            limit = UNSET
        else:
            limit = V0043Uint32NoValStruct.from_dict(_limit)

        _total = d.pop("total", UNSET)
        total: Union[Unset, V0043JobTimeTotal]
        if isinstance(_total, Unset):
            total = UNSET
        else:
            total = V0043JobTimeTotal.from_dict(_total)

        _user = d.pop("user", UNSET)
        user: Union[Unset, V0043JobTimeUser]
        if isinstance(_user, Unset):
            user = UNSET
        else:
            user = V0043JobTimeUser.from_dict(_user)

        v0043_job_time = cls(
            elapsed=elapsed,
            eligible=eligible,
            end=end,
            planned=planned,
            start=start,
            submission=submission,
            suspended=suspended,
            system=system,
            limit=limit,
            total=total,
            user=user,
        )

        v0043_job_time.additional_properties = d
        return v0043_job_time

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
