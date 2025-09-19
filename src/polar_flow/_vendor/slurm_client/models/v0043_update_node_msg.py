from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_update_node_msg_state_item import V0043UpdateNodeMsgStateItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043UpdateNodeMsg")


@_attrs_define
class V0043UpdateNodeMsg:
    comment: Union[Unset, str] = UNSET
    """ Arbitrary comment """
    cpu_bind: Union[Unset, int] = UNSET
    """ Default method for binding tasks to allocated CPUs """
    extra: Union[Unset, str] = UNSET
    """ Arbitrary string used for node filtering if extra constraints are enabled """
    features: Union[Unset, list[str]] = UNSET
    features_act: Union[Unset, list[str]] = UNSET
    gres: Union[Unset, str] = UNSET
    """ Generic resources """
    address: Union[Unset, list[str]] = UNSET
    hostname: Union[Unset, list[str]] = UNSET
    name: Union[Unset, list[str]] = UNSET
    state: Union[Unset, list[V0043UpdateNodeMsgStateItem]] = UNSET
    """ New state to assign to the node """
    reason: Union[Unset, str] = UNSET
    """ Reason for node being DOWN or DRAINING """
    reason_uid: Union[Unset, str] = UNSET
    """ User ID to associate with the reason (needed if user root is sending message) """
    resume_after: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    topology_str: Union[Unset, str] = UNSET
    """ Topology """
    weight: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        comment = self.comment

        cpu_bind = self.cpu_bind

        extra = self.extra

        features: Union[Unset, list[str]] = UNSET
        if not isinstance(self.features, Unset):
            features = self.features

        features_act: Union[Unset, list[str]] = UNSET
        if not isinstance(self.features_act, Unset):
            features_act = self.features_act

        gres = self.gres

        address: Union[Unset, list[str]] = UNSET
        if not isinstance(self.address, Unset):
            address = self.address

        hostname: Union[Unset, list[str]] = UNSET
        if not isinstance(self.hostname, Unset):
            hostname = self.hostname

        name: Union[Unset, list[str]] = UNSET
        if not isinstance(self.name, Unset):
            name = self.name

        state: Union[Unset, list[str]] = UNSET
        if not isinstance(self.state, Unset):
            state = []
            for state_item_data in self.state:
                state_item = state_item_data.value
                state.append(state_item)

        reason = self.reason

        reason_uid = self.reason_uid

        resume_after: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.resume_after, Unset):
            resume_after = self.resume_after.to_dict()

        topology_str = self.topology_str

        weight: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.weight, Unset):
            weight = self.weight.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if comment is not UNSET:
            field_dict["comment"] = comment
        if cpu_bind is not UNSET:
            field_dict["cpu_bind"] = cpu_bind
        if extra is not UNSET:
            field_dict["extra"] = extra
        if features is not UNSET:
            field_dict["features"] = features
        if features_act is not UNSET:
            field_dict["features_act"] = features_act
        if gres is not UNSET:
            field_dict["gres"] = gres
        if address is not UNSET:
            field_dict["address"] = address
        if hostname is not UNSET:
            field_dict["hostname"] = hostname
        if name is not UNSET:
            field_dict["name"] = name
        if state is not UNSET:
            field_dict["state"] = state
        if reason is not UNSET:
            field_dict["reason"] = reason
        if reason_uid is not UNSET:
            field_dict["reason_uid"] = reason_uid
        if resume_after is not UNSET:
            field_dict["resume_after"] = resume_after
        if topology_str is not UNSET:
            field_dict["topology_str"] = topology_str
        if weight is not UNSET:
            field_dict["weight"] = weight

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        comment = d.pop("comment", UNSET)

        cpu_bind = d.pop("cpu_bind", UNSET)

        extra = d.pop("extra", UNSET)

        features = cast(list[str], d.pop("features", UNSET))

        features_act = cast(list[str], d.pop("features_act", UNSET))

        gres = d.pop("gres", UNSET)

        address = cast(list[str], d.pop("address", UNSET))

        hostname = cast(list[str], d.pop("hostname", UNSET))

        name = cast(list[str], d.pop("name", UNSET))

        state = []
        _state = d.pop("state", UNSET)
        for state_item_data in _state or []:
            state_item = V0043UpdateNodeMsgStateItem(state_item_data)

            state.append(state_item)

        reason = d.pop("reason", UNSET)

        reason_uid = d.pop("reason_uid", UNSET)

        _resume_after = d.pop("resume_after", UNSET)
        resume_after: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_resume_after, Unset):
            resume_after = UNSET
        else:
            resume_after = V0043Uint32NoValStruct.from_dict(_resume_after)

        topology_str = d.pop("topology_str", UNSET)

        _weight = d.pop("weight", UNSET)
        weight: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_weight, Unset):
            weight = UNSET
        else:
            weight = V0043Uint32NoValStruct.from_dict(_weight)

        v0043_update_node_msg = cls(
            comment=comment,
            cpu_bind=cpu_bind,
            extra=extra,
            features=features,
            features_act=features_act,
            gres=gres,
            address=address,
            hostname=hostname,
            name=name,
            state=state,
            reason=reason,
            reason_uid=reason_uid,
            resume_after=resume_after,
            topology_str=topology_str,
            weight=weight,
        )

        v0043_update_node_msg.additional_properties = d
        return v0043_update_node_msg

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
