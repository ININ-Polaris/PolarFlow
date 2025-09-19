from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_instance_time import V0043InstanceTime


T = TypeVar("T", bound="V0043Instance")


@_attrs_define
class V0043Instance:
    cluster: Union[Unset, str] = UNSET
    """ Cluster name """
    extra: Union[Unset, str] = UNSET
    """ Arbitrary string used for node filtering if extra constraints are enabled """
    instance_id: Union[Unset, str] = UNSET
    """ Cloud instance ID """
    instance_type: Union[Unset, str] = UNSET
    """ Cloud instance type """
    node_name: Union[Unset, str] = UNSET
    """ NodeName """
    time: Union[Unset, "V0043InstanceTime"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cluster = self.cluster

        extra = self.extra

        instance_id = self.instance_id

        instance_type = self.instance_type

        node_name = self.node_name

        time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.time, Unset):
            time = self.time.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if cluster is not UNSET:
            field_dict["cluster"] = cluster
        if extra is not UNSET:
            field_dict["extra"] = extra
        if instance_id is not UNSET:
            field_dict["instance_id"] = instance_id
        if instance_type is not UNSET:
            field_dict["instance_type"] = instance_type
        if node_name is not UNSET:
            field_dict["node_name"] = node_name
        if time is not UNSET:
            field_dict["time"] = time

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_instance_time import V0043InstanceTime

        d = dict(src_dict)
        cluster = d.pop("cluster", UNSET)

        extra = d.pop("extra", UNSET)

        instance_id = d.pop("instance_id", UNSET)

        instance_type = d.pop("instance_type", UNSET)

        node_name = d.pop("node_name", UNSET)

        _time = d.pop("time", UNSET)
        time: Union[Unset, V0043InstanceTime]
        if isinstance(_time, Unset):
            time = UNSET
        else:
            time = V0043InstanceTime.from_dict(_time)

        v0043_instance = cls(
            cluster=cluster,
            extra=extra,
            instance_id=instance_id,
            instance_type=instance_type,
            node_name=node_name,
            time=time,
        )

        v0043_instance.additional_properties = d
        return v0043_instance

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
