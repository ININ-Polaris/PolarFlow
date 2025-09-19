from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_user_administrator_level_item import V0043UserAdministratorLevelItem
from ..models.v0043_user_flags_item import V0043UserFlagsItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_assoc_short import V0043AssocShort
    from ..models.v0043_coord import V0043Coord
    from ..models.v0043_user_default import V0043UserDefault
    from ..models.v0043_wckey import V0043Wckey


T = TypeVar("T", bound="V0043User")


@_attrs_define
class V0043User:
    """
    Attributes:
        name (str): User name
        administrator_level (Union[Unset, list[V0043UserAdministratorLevelItem]]): AdminLevel granted to the user
        associations (Union[Unset, list['V0043AssocShort']]):
        coordinators (Union[Unset, list['V0043Coord']]):
        default (Union[Unset, V0043UserDefault]):
        flags (Union[Unset, list[V0043UserFlagsItem]]): Flags associated with this user
        old_name (Union[Unset, str]): Previous user name
        wckeys (Union[Unset, list['V0043Wckey']]):
    """

    name: str
    administrator_level: Union[Unset, list[V0043UserAdministratorLevelItem]] = UNSET
    associations: Union[Unset, list["V0043AssocShort"]] = UNSET
    coordinators: Union[Unset, list["V0043Coord"]] = UNSET
    default: Union[Unset, "V0043UserDefault"] = UNSET
    flags: Union[Unset, list[V0043UserFlagsItem]] = UNSET
    old_name: Union[Unset, str] = UNSET
    wckeys: Union[Unset, list["V0043Wckey"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        administrator_level: Union[Unset, list[str]] = UNSET
        if not isinstance(self.administrator_level, Unset):
            administrator_level = []
            for administrator_level_item_data in self.administrator_level:
                administrator_level_item = administrator_level_item_data.value
                administrator_level.append(administrator_level_item)

        associations: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.associations, Unset):
            associations = []
            for componentsschemasv0_0_43_assoc_short_list_item_data in self.associations:
                componentsschemasv0_0_43_assoc_short_list_item = (
                    componentsschemasv0_0_43_assoc_short_list_item_data.to_dict()
                )
                associations.append(componentsschemasv0_0_43_assoc_short_list_item)

        coordinators: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.coordinators, Unset):
            coordinators = []
            for componentsschemasv0_0_43_coord_list_item_data in self.coordinators:
                componentsschemasv0_0_43_coord_list_item = componentsschemasv0_0_43_coord_list_item_data.to_dict()
                coordinators.append(componentsschemasv0_0_43_coord_list_item)

        default: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.default, Unset):
            default = self.default.to_dict()

        flags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.flags, Unset):
            flags = []
            for flags_item_data in self.flags:
                flags_item = flags_item_data.value
                flags.append(flags_item)

        old_name = self.old_name

        wckeys: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.wckeys, Unset):
            wckeys = []
            for componentsschemasv0_0_43_wckey_list_item_data in self.wckeys:
                componentsschemasv0_0_43_wckey_list_item = componentsschemasv0_0_43_wckey_list_item_data.to_dict()
                wckeys.append(componentsschemasv0_0_43_wckey_list_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if administrator_level is not UNSET:
            field_dict["administrator_level"] = administrator_level
        if associations is not UNSET:
            field_dict["associations"] = associations
        if coordinators is not UNSET:
            field_dict["coordinators"] = coordinators
        if default is not UNSET:
            field_dict["default"] = default
        if flags is not UNSET:
            field_dict["flags"] = flags
        if old_name is not UNSET:
            field_dict["old_name"] = old_name
        if wckeys is not UNSET:
            field_dict["wckeys"] = wckeys

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_assoc_short import V0043AssocShort
        from ..models.v0043_coord import V0043Coord
        from ..models.v0043_user_default import V0043UserDefault
        from ..models.v0043_wckey import V0043Wckey

        d = dict(src_dict)
        name = d.pop("name")

        administrator_level = []
        _administrator_level = d.pop("administrator_level", UNSET)
        for administrator_level_item_data in _administrator_level or []:
            administrator_level_item = V0043UserAdministratorLevelItem(administrator_level_item_data)

            administrator_level.append(administrator_level_item)

        associations = []
        _associations = d.pop("associations", UNSET)
        for componentsschemasv0_0_43_assoc_short_list_item_data in _associations or []:
            componentsschemasv0_0_43_assoc_short_list_item = V0043AssocShort.from_dict(
                componentsschemasv0_0_43_assoc_short_list_item_data
            )

            associations.append(componentsschemasv0_0_43_assoc_short_list_item)

        coordinators = []
        _coordinators = d.pop("coordinators", UNSET)
        for componentsschemasv0_0_43_coord_list_item_data in _coordinators or []:
            componentsschemasv0_0_43_coord_list_item = V0043Coord.from_dict(
                componentsschemasv0_0_43_coord_list_item_data
            )

            coordinators.append(componentsschemasv0_0_43_coord_list_item)

        _default = d.pop("default", UNSET)
        default: Union[Unset, V0043UserDefault]
        if isinstance(_default, Unset):
            default = UNSET
        else:
            default = V0043UserDefault.from_dict(_default)

        flags = []
        _flags = d.pop("flags", UNSET)
        for flags_item_data in _flags or []:
            flags_item = V0043UserFlagsItem(flags_item_data)

            flags.append(flags_item)

        old_name = d.pop("old_name", UNSET)

        wckeys = []
        _wckeys = d.pop("wckeys", UNSET)
        for componentsschemasv0_0_43_wckey_list_item_data in _wckeys or []:
            componentsschemasv0_0_43_wckey_list_item = V0043Wckey.from_dict(
                componentsschemasv0_0_43_wckey_list_item_data
            )

            wckeys.append(componentsschemasv0_0_43_wckey_list_item)

        v0043_user = cls(
            name=name,
            administrator_level=administrator_level,
            associations=associations,
            coordinators=coordinators,
            default=default,
            flags=flags,
            old_name=old_name,
            wckeys=wckeys,
        )

        v0043_user.additional_properties = d
        return v0043_user

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
