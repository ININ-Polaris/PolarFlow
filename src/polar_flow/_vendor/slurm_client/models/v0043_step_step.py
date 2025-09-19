from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="V0043StepStep")


@_attrs_define
class V0043StepStep:
    id: Union[Unset, str] = UNSET
    """ Step ID (Slurm job step ID) """
    name: Union[Unset, str] = UNSET
    """ Step name """
    stderr: Union[Unset, str] = UNSET
    """ Path to stderr file """
    stdin: Union[Unset, str] = UNSET
    """ Path to stdin file """
    stdout: Union[Unset, str] = UNSET
    """ Path to stdout file """
    stderr_expanded: Union[Unset, str] = UNSET
    """ Step stderr with expanded fields """
    stdin_expanded: Union[Unset, str] = UNSET
    """ Step stdin with expanded fields """
    stdout_expanded: Union[Unset, str] = UNSET
    """ Step stdout with expanded fields """
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        stderr = self.stderr

        stdin = self.stdin

        stdout = self.stdout

        stderr_expanded = self.stderr_expanded

        stdin_expanded = self.stdin_expanded

        stdout_expanded = self.stdout_expanded

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name
        if stderr is not UNSET:
            field_dict["stderr"] = stderr
        if stdin is not UNSET:
            field_dict["stdin"] = stdin
        if stdout is not UNSET:
            field_dict["stdout"] = stdout
        if stderr_expanded is not UNSET:
            field_dict["stderr_expanded"] = stderr_expanded
        if stdin_expanded is not UNSET:
            field_dict["stdin_expanded"] = stdin_expanded
        if stdout_expanded is not UNSET:
            field_dict["stdout_expanded"] = stdout_expanded

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        stderr = d.pop("stderr", UNSET)

        stdin = d.pop("stdin", UNSET)

        stdout = d.pop("stdout", UNSET)

        stderr_expanded = d.pop("stderr_expanded", UNSET)

        stdin_expanded = d.pop("stdin_expanded", UNSET)

        stdout_expanded = d.pop("stdout_expanded", UNSET)

        v0043_step_step = cls(
            id=id,
            name=name,
            stderr=stderr,
            stdin=stdin,
            stdout=stdout,
            stderr_expanded=stderr_expanded,
            stdin_expanded=stdin_expanded,
            stdout_expanded=stdout_expanded,
        )

        v0043_step_step.additional_properties = d
        return v0043_step_step

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
