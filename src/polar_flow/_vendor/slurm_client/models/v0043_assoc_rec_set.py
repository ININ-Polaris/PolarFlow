from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_tres import V0043Tres
    from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct


T = TypeVar("T", bound="V0043AssocRecSet")


@_attrs_define
class V0043AssocRecSet:
    """
    Attributes:
        comment (Union[Unset, str]): Arbitrary comment
        defaultqos (Union[Unset, str]): Default QOS
        grpjobs (Union[Unset, V0043Uint32NoValStruct]):
        grpjobsaccrue (Union[Unset, V0043Uint32NoValStruct]):
        grpsubmitjobs (Union[Unset, V0043Uint32NoValStruct]):
        grptres (Union[Unset, list['V0043Tres']]):
        grptresmins (Union[Unset, list['V0043Tres']]):
        grptresrunmins (Union[Unset, list['V0043Tres']]):
        grpwall (Union[Unset, V0043Uint32NoValStruct]):
        maxjobs (Union[Unset, V0043Uint32NoValStruct]):
        maxjobsaccrue (Union[Unset, V0043Uint32NoValStruct]):
        maxsubmitjobs (Union[Unset, V0043Uint32NoValStruct]):
        maxtresminsperjob (Union[Unset, list['V0043Tres']]):
        maxtresrunmins (Union[Unset, list['V0043Tres']]):
        maxtresperjob (Union[Unset, list['V0043Tres']]):
        maxtrespernode (Union[Unset, list['V0043Tres']]):
        maxwalldurationperjob (Union[Unset, V0043Uint32NoValStruct]):
        minpriothresh (Union[Unset, V0043Uint32NoValStruct]):
        parent (Union[Unset, str]): Name of parent account
        priority (Union[Unset, V0043Uint32NoValStruct]):
        qoslevel (Union[Unset, list[str]]): List of QOS names
        fairshare (Union[Unset, int]): Allocated shares used for fairshare calculation
    """

    comment: Union[Unset, str] = UNSET
    defaultqos: Union[Unset, str] = UNSET
    grpjobs: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    grpjobsaccrue: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    grpsubmitjobs: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    grptres: Union[Unset, list["V0043Tres"]] = UNSET
    grptresmins: Union[Unset, list["V0043Tres"]] = UNSET
    grptresrunmins: Union[Unset, list["V0043Tres"]] = UNSET
    grpwall: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    maxjobs: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    maxjobsaccrue: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    maxsubmitjobs: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    maxtresminsperjob: Union[Unset, list["V0043Tres"]] = UNSET
    maxtresrunmins: Union[Unset, list["V0043Tres"]] = UNSET
    maxtresperjob: Union[Unset, list["V0043Tres"]] = UNSET
    maxtrespernode: Union[Unset, list["V0043Tres"]] = UNSET
    maxwalldurationperjob: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    minpriothresh: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    parent: Union[Unset, str] = UNSET
    priority: Union[Unset, "V0043Uint32NoValStruct"] = UNSET
    qoslevel: Union[Unset, list[str]] = UNSET
    fairshare: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        comment = self.comment

        defaultqos = self.defaultqos

        grpjobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.grpjobs, Unset):
            grpjobs = self.grpjobs.to_dict()

        grpjobsaccrue: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.grpjobsaccrue, Unset):
            grpjobsaccrue = self.grpjobsaccrue.to_dict()

        grpsubmitjobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.grpsubmitjobs, Unset):
            grpsubmitjobs = self.grpsubmitjobs.to_dict()

        grptres: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.grptres, Unset):
            grptres = []
            for componentsschemasv0_0_43_tres_list_item_data in self.grptres:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                grptres.append(componentsschemasv0_0_43_tres_list_item)

        grptresmins: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.grptresmins, Unset):
            grptresmins = []
            for componentsschemasv0_0_43_tres_list_item_data in self.grptresmins:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                grptresmins.append(componentsschemasv0_0_43_tres_list_item)

        grptresrunmins: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.grptresrunmins, Unset):
            grptresrunmins = []
            for componentsschemasv0_0_43_tres_list_item_data in self.grptresrunmins:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                grptresrunmins.append(componentsschemasv0_0_43_tres_list_item)

        grpwall: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.grpwall, Unset):
            grpwall = self.grpwall.to_dict()

        maxjobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.maxjobs, Unset):
            maxjobs = self.maxjobs.to_dict()

        maxjobsaccrue: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.maxjobsaccrue, Unset):
            maxjobsaccrue = self.maxjobsaccrue.to_dict()

        maxsubmitjobs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.maxsubmitjobs, Unset):
            maxsubmitjobs = self.maxsubmitjobs.to_dict()

        maxtresminsperjob: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.maxtresminsperjob, Unset):
            maxtresminsperjob = []
            for componentsschemasv0_0_43_tres_list_item_data in self.maxtresminsperjob:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                maxtresminsperjob.append(componentsschemasv0_0_43_tres_list_item)

        maxtresrunmins: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.maxtresrunmins, Unset):
            maxtresrunmins = []
            for componentsschemasv0_0_43_tres_list_item_data in self.maxtresrunmins:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                maxtresrunmins.append(componentsschemasv0_0_43_tres_list_item)

        maxtresperjob: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.maxtresperjob, Unset):
            maxtresperjob = []
            for componentsschemasv0_0_43_tres_list_item_data in self.maxtresperjob:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                maxtresperjob.append(componentsschemasv0_0_43_tres_list_item)

        maxtrespernode: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.maxtrespernode, Unset):
            maxtrespernode = []
            for componentsschemasv0_0_43_tres_list_item_data in self.maxtrespernode:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                maxtrespernode.append(componentsschemasv0_0_43_tres_list_item)

        maxwalldurationperjob: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.maxwalldurationperjob, Unset):
            maxwalldurationperjob = self.maxwalldurationperjob.to_dict()

        minpriothresh: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.minpriothresh, Unset):
            minpriothresh = self.minpriothresh.to_dict()

        parent = self.parent

        priority: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.priority, Unset):
            priority = self.priority.to_dict()

        qoslevel: Union[Unset, list[str]] = UNSET
        if not isinstance(self.qoslevel, Unset):
            qoslevel = self.qoslevel

        fairshare = self.fairshare

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if comment is not UNSET:
            field_dict["comment"] = comment
        if defaultqos is not UNSET:
            field_dict["defaultqos"] = defaultqos
        if grpjobs is not UNSET:
            field_dict["grpjobs"] = grpjobs
        if grpjobsaccrue is not UNSET:
            field_dict["grpjobsaccrue"] = grpjobsaccrue
        if grpsubmitjobs is not UNSET:
            field_dict["grpsubmitjobs"] = grpsubmitjobs
        if grptres is not UNSET:
            field_dict["grptres"] = grptres
        if grptresmins is not UNSET:
            field_dict["grptresmins"] = grptresmins
        if grptresrunmins is not UNSET:
            field_dict["grptresrunmins"] = grptresrunmins
        if grpwall is not UNSET:
            field_dict["grpwall"] = grpwall
        if maxjobs is not UNSET:
            field_dict["maxjobs"] = maxjobs
        if maxjobsaccrue is not UNSET:
            field_dict["maxjobsaccrue"] = maxjobsaccrue
        if maxsubmitjobs is not UNSET:
            field_dict["maxsubmitjobs"] = maxsubmitjobs
        if maxtresminsperjob is not UNSET:
            field_dict["maxtresminsperjob"] = maxtresminsperjob
        if maxtresrunmins is not UNSET:
            field_dict["maxtresrunmins"] = maxtresrunmins
        if maxtresperjob is not UNSET:
            field_dict["maxtresperjob"] = maxtresperjob
        if maxtrespernode is not UNSET:
            field_dict["maxtrespernode"] = maxtrespernode
        if maxwalldurationperjob is not UNSET:
            field_dict["maxwalldurationperjob"] = maxwalldurationperjob
        if minpriothresh is not UNSET:
            field_dict["minpriothresh"] = minpriothresh
        if parent is not UNSET:
            field_dict["parent"] = parent
        if priority is not UNSET:
            field_dict["priority"] = priority
        if qoslevel is not UNSET:
            field_dict["qoslevel"] = qoslevel
        if fairshare is not UNSET:
            field_dict["fairshare"] = fairshare

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_tres import V0043Tres
        from ..models.v0043_uint_32_no_val_struct import V0043Uint32NoValStruct

        d = dict(src_dict)
        comment = d.pop("comment", UNSET)

        defaultqos = d.pop("defaultqos", UNSET)

        _grpjobs = d.pop("grpjobs", UNSET)
        grpjobs: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_grpjobs, Unset):
            grpjobs = UNSET
        else:
            grpjobs = V0043Uint32NoValStruct.from_dict(_grpjobs)

        _grpjobsaccrue = d.pop("grpjobsaccrue", UNSET)
        grpjobsaccrue: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_grpjobsaccrue, Unset):
            grpjobsaccrue = UNSET
        else:
            grpjobsaccrue = V0043Uint32NoValStruct.from_dict(_grpjobsaccrue)

        _grpsubmitjobs = d.pop("grpsubmitjobs", UNSET)
        grpsubmitjobs: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_grpsubmitjobs, Unset):
            grpsubmitjobs = UNSET
        else:
            grpsubmitjobs = V0043Uint32NoValStruct.from_dict(_grpsubmitjobs)

        grptres = []
        _grptres = d.pop("grptres", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _grptres or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            grptres.append(componentsschemasv0_0_43_tres_list_item)

        grptresmins = []
        _grptresmins = d.pop("grptresmins", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _grptresmins or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            grptresmins.append(componentsschemasv0_0_43_tres_list_item)

        grptresrunmins = []
        _grptresrunmins = d.pop("grptresrunmins", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _grptresrunmins or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            grptresrunmins.append(componentsschemasv0_0_43_tres_list_item)

        _grpwall = d.pop("grpwall", UNSET)
        grpwall: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_grpwall, Unset):
            grpwall = UNSET
        else:
            grpwall = V0043Uint32NoValStruct.from_dict(_grpwall)

        _maxjobs = d.pop("maxjobs", UNSET)
        maxjobs: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_maxjobs, Unset):
            maxjobs = UNSET
        else:
            maxjobs = V0043Uint32NoValStruct.from_dict(_maxjobs)

        _maxjobsaccrue = d.pop("maxjobsaccrue", UNSET)
        maxjobsaccrue: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_maxjobsaccrue, Unset):
            maxjobsaccrue = UNSET
        else:
            maxjobsaccrue = V0043Uint32NoValStruct.from_dict(_maxjobsaccrue)

        _maxsubmitjobs = d.pop("maxsubmitjobs", UNSET)
        maxsubmitjobs: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_maxsubmitjobs, Unset):
            maxsubmitjobs = UNSET
        else:
            maxsubmitjobs = V0043Uint32NoValStruct.from_dict(_maxsubmitjobs)

        maxtresminsperjob = []
        _maxtresminsperjob = d.pop("maxtresminsperjob", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _maxtresminsperjob or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            maxtresminsperjob.append(componentsschemasv0_0_43_tres_list_item)

        maxtresrunmins = []
        _maxtresrunmins = d.pop("maxtresrunmins", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _maxtresrunmins or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            maxtresrunmins.append(componentsschemasv0_0_43_tres_list_item)

        maxtresperjob = []
        _maxtresperjob = d.pop("maxtresperjob", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _maxtresperjob or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            maxtresperjob.append(componentsschemasv0_0_43_tres_list_item)

        maxtrespernode = []
        _maxtrespernode = d.pop("maxtrespernode", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _maxtrespernode or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            maxtrespernode.append(componentsschemasv0_0_43_tres_list_item)

        _maxwalldurationperjob = d.pop("maxwalldurationperjob", UNSET)
        maxwalldurationperjob: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_maxwalldurationperjob, Unset):
            maxwalldurationperjob = UNSET
        else:
            maxwalldurationperjob = V0043Uint32NoValStruct.from_dict(_maxwalldurationperjob)

        _minpriothresh = d.pop("minpriothresh", UNSET)
        minpriothresh: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_minpriothresh, Unset):
            minpriothresh = UNSET
        else:
            minpriothresh = V0043Uint32NoValStruct.from_dict(_minpriothresh)

        parent = d.pop("parent", UNSET)

        _priority = d.pop("priority", UNSET)
        priority: Union[Unset, V0043Uint32NoValStruct]
        if isinstance(_priority, Unset):
            priority = UNSET
        else:
            priority = V0043Uint32NoValStruct.from_dict(_priority)

        qoslevel = cast(list[str], d.pop("qoslevel", UNSET))

        fairshare = d.pop("fairshare", UNSET)

        v0043_assoc_rec_set = cls(
            comment=comment,
            defaultqos=defaultqos,
            grpjobs=grpjobs,
            grpjobsaccrue=grpjobsaccrue,
            grpsubmitjobs=grpsubmitjobs,
            grptres=grptres,
            grptresmins=grptresmins,
            grptresrunmins=grptresrunmins,
            grpwall=grpwall,
            maxjobs=maxjobs,
            maxjobsaccrue=maxjobsaccrue,
            maxsubmitjobs=maxsubmitjobs,
            maxtresminsperjob=maxtresminsperjob,
            maxtresrunmins=maxtresrunmins,
            maxtresperjob=maxtresperjob,
            maxtrespernode=maxtrespernode,
            maxwalldurationperjob=maxwalldurationperjob,
            minpriothresh=minpriothresh,
            parent=parent,
            priority=priority,
            qoslevel=qoslevel,
            fairshare=fairshare,
        )

        v0043_assoc_rec_set.additional_properties = d
        return v0043_assoc_rec_set

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
