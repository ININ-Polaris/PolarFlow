from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_account import V0043Account
    from ..models.v0043_assoc import V0043Assoc
    from ..models.v0043_cluster_rec import V0043ClusterRec
    from ..models.v0043_instance import V0043Instance
    from ..models.v0043_openapi_error import V0043OpenapiError
    from ..models.v0043_openapi_meta import V0043OpenapiMeta
    from ..models.v0043_openapi_warning import V0043OpenapiWarning
    from ..models.v0043_qos import V0043Qos
    from ..models.v0043_tres import V0043Tres
    from ..models.v0043_user import V0043User
    from ..models.v0043_wckey import V0043Wckey


T = TypeVar("T", bound="V0043OpenapiSlurmdbdConfigResp")


@_attrs_define
class V0043OpenapiSlurmdbdConfigResp:
    """
    Attributes:
        clusters (Union[Unset, list['V0043ClusterRec']]):
        tres (Union[Unset, list['V0043Tres']]):
        accounts (Union[Unset, list['V0043Account']]):
        users (Union[Unset, list['V0043User']]):
        qos (Union[Unset, list['V0043Qos']]):
        wckeys (Union[Unset, list['V0043Wckey']]):
        associations (Union[Unset, list['V0043Assoc']]):
        instances (Union[Unset, list['V0043Instance']]):
        meta (Union[Unset, V0043OpenapiMeta]):
        errors (Union[Unset, list['V0043OpenapiError']]):
        warnings (Union[Unset, list['V0043OpenapiWarning']]):
    """

    clusters: Union[Unset, list["V0043ClusterRec"]] = UNSET
    tres: Union[Unset, list["V0043Tres"]] = UNSET
    accounts: Union[Unset, list["V0043Account"]] = UNSET
    users: Union[Unset, list["V0043User"]] = UNSET
    qos: Union[Unset, list["V0043Qos"]] = UNSET
    wckeys: Union[Unset, list["V0043Wckey"]] = UNSET
    associations: Union[Unset, list["V0043Assoc"]] = UNSET
    instances: Union[Unset, list["V0043Instance"]] = UNSET
    meta: Union[Unset, "V0043OpenapiMeta"] = UNSET
    errors: Union[Unset, list["V0043OpenapiError"]] = UNSET
    warnings: Union[Unset, list["V0043OpenapiWarning"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        clusters: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.clusters, Unset):
            clusters = []
            for componentsschemasv0_0_43_cluster_rec_list_item_data in self.clusters:
                componentsschemasv0_0_43_cluster_rec_list_item = (
                    componentsschemasv0_0_43_cluster_rec_list_item_data.to_dict()
                )
                clusters.append(componentsschemasv0_0_43_cluster_rec_list_item)

        tres: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = []
            for componentsschemasv0_0_43_tres_list_item_data in self.tres:
                componentsschemasv0_0_43_tres_list_item = componentsschemasv0_0_43_tres_list_item_data.to_dict()
                tres.append(componentsschemasv0_0_43_tres_list_item)

        accounts: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.accounts, Unset):
            accounts = []
            for componentsschemasv0_0_43_account_list_item_data in self.accounts:
                componentsschemasv0_0_43_account_list_item = componentsschemasv0_0_43_account_list_item_data.to_dict()
                accounts.append(componentsschemasv0_0_43_account_list_item)

        users: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.users, Unset):
            users = []
            for componentsschemasv0_0_43_user_list_item_data in self.users:
                componentsschemasv0_0_43_user_list_item = componentsschemasv0_0_43_user_list_item_data.to_dict()
                users.append(componentsschemasv0_0_43_user_list_item)

        qos: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.qos, Unset):
            qos = []
            for componentsschemasv0_0_43_qos_list_item_data in self.qos:
                componentsschemasv0_0_43_qos_list_item = componentsschemasv0_0_43_qos_list_item_data.to_dict()
                qos.append(componentsschemasv0_0_43_qos_list_item)

        wckeys: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.wckeys, Unset):
            wckeys = []
            for componentsschemasv0_0_43_wckey_list_item_data in self.wckeys:
                componentsschemasv0_0_43_wckey_list_item = componentsschemasv0_0_43_wckey_list_item_data.to_dict()
                wckeys.append(componentsschemasv0_0_43_wckey_list_item)

        associations: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.associations, Unset):
            associations = []
            for componentsschemasv0_0_43_assoc_list_item_data in self.associations:
                componentsschemasv0_0_43_assoc_list_item = componentsschemasv0_0_43_assoc_list_item_data.to_dict()
                associations.append(componentsschemasv0_0_43_assoc_list_item)

        instances: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.instances, Unset):
            instances = []
            for componentsschemasv0_0_43_instance_list_item_data in self.instances:
                componentsschemasv0_0_43_instance_list_item = componentsschemasv0_0_43_instance_list_item_data.to_dict()
                instances.append(componentsschemasv0_0_43_instance_list_item)

        meta: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.meta, Unset):
            meta = self.meta.to_dict()

        errors: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.errors, Unset):
            errors = []
            for componentsschemasv0_0_43_openapi_errors_item_data in self.errors:
                componentsschemasv0_0_43_openapi_errors_item = (
                    componentsschemasv0_0_43_openapi_errors_item_data.to_dict()
                )
                errors.append(componentsschemasv0_0_43_openapi_errors_item)

        warnings: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.warnings, Unset):
            warnings = []
            for componentsschemasv0_0_43_openapi_warnings_item_data in self.warnings:
                componentsschemasv0_0_43_openapi_warnings_item = (
                    componentsschemasv0_0_43_openapi_warnings_item_data.to_dict()
                )
                warnings.append(componentsschemasv0_0_43_openapi_warnings_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if clusters is not UNSET:
            field_dict["clusters"] = clusters
        if tres is not UNSET:
            field_dict["tres"] = tres
        if accounts is not UNSET:
            field_dict["accounts"] = accounts
        if users is not UNSET:
            field_dict["users"] = users
        if qos is not UNSET:
            field_dict["qos"] = qos
        if wckeys is not UNSET:
            field_dict["wckeys"] = wckeys
        if associations is not UNSET:
            field_dict["associations"] = associations
        if instances is not UNSET:
            field_dict["instances"] = instances
        if meta is not UNSET:
            field_dict["meta"] = meta
        if errors is not UNSET:
            field_dict["errors"] = errors
        if warnings is not UNSET:
            field_dict["warnings"] = warnings

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_account import V0043Account
        from ..models.v0043_assoc import V0043Assoc
        from ..models.v0043_cluster_rec import V0043ClusterRec
        from ..models.v0043_instance import V0043Instance
        from ..models.v0043_openapi_error import V0043OpenapiError
        from ..models.v0043_openapi_meta import V0043OpenapiMeta
        from ..models.v0043_openapi_warning import V0043OpenapiWarning
        from ..models.v0043_qos import V0043Qos
        from ..models.v0043_tres import V0043Tres
        from ..models.v0043_user import V0043User
        from ..models.v0043_wckey import V0043Wckey

        d = dict(src_dict)
        clusters = []
        _clusters = d.pop("clusters", UNSET)
        for componentsschemasv0_0_43_cluster_rec_list_item_data in _clusters or []:
            componentsschemasv0_0_43_cluster_rec_list_item = V0043ClusterRec.from_dict(
                componentsschemasv0_0_43_cluster_rec_list_item_data
            )

            clusters.append(componentsschemasv0_0_43_cluster_rec_list_item)

        tres = []
        _tres = d.pop("tres", UNSET)
        for componentsschemasv0_0_43_tres_list_item_data in _tres or []:
            componentsschemasv0_0_43_tres_list_item = V0043Tres.from_dict(componentsschemasv0_0_43_tres_list_item_data)

            tres.append(componentsschemasv0_0_43_tres_list_item)

        accounts = []
        _accounts = d.pop("accounts", UNSET)
        for componentsschemasv0_0_43_account_list_item_data in _accounts or []:
            componentsschemasv0_0_43_account_list_item = V0043Account.from_dict(
                componentsschemasv0_0_43_account_list_item_data
            )

            accounts.append(componentsschemasv0_0_43_account_list_item)

        users = []
        _users = d.pop("users", UNSET)
        for componentsschemasv0_0_43_user_list_item_data in _users or []:
            componentsschemasv0_0_43_user_list_item = V0043User.from_dict(componentsschemasv0_0_43_user_list_item_data)

            users.append(componentsschemasv0_0_43_user_list_item)

        qos = []
        _qos = d.pop("qos", UNSET)
        for componentsschemasv0_0_43_qos_list_item_data in _qos or []:
            componentsschemasv0_0_43_qos_list_item = V0043Qos.from_dict(componentsschemasv0_0_43_qos_list_item_data)

            qos.append(componentsschemasv0_0_43_qos_list_item)

        wckeys = []
        _wckeys = d.pop("wckeys", UNSET)
        for componentsschemasv0_0_43_wckey_list_item_data in _wckeys or []:
            componentsschemasv0_0_43_wckey_list_item = V0043Wckey.from_dict(
                componentsschemasv0_0_43_wckey_list_item_data
            )

            wckeys.append(componentsschemasv0_0_43_wckey_list_item)

        associations = []
        _associations = d.pop("associations", UNSET)
        for componentsschemasv0_0_43_assoc_list_item_data in _associations or []:
            componentsschemasv0_0_43_assoc_list_item = V0043Assoc.from_dict(
                componentsschemasv0_0_43_assoc_list_item_data
            )

            associations.append(componentsschemasv0_0_43_assoc_list_item)

        instances = []
        _instances = d.pop("instances", UNSET)
        for componentsschemasv0_0_43_instance_list_item_data in _instances or []:
            componentsschemasv0_0_43_instance_list_item = V0043Instance.from_dict(
                componentsschemasv0_0_43_instance_list_item_data
            )

            instances.append(componentsschemasv0_0_43_instance_list_item)

        _meta = d.pop("meta", UNSET)
        meta: Union[Unset, V0043OpenapiMeta]
        if isinstance(_meta, Unset):
            meta = UNSET
        else:
            meta = V0043OpenapiMeta.from_dict(_meta)

        errors = []
        _errors = d.pop("errors", UNSET)
        for componentsschemasv0_0_43_openapi_errors_item_data in _errors or []:
            componentsschemasv0_0_43_openapi_errors_item = V0043OpenapiError.from_dict(
                componentsschemasv0_0_43_openapi_errors_item_data
            )

            errors.append(componentsschemasv0_0_43_openapi_errors_item)

        warnings = []
        _warnings = d.pop("warnings", UNSET)
        for componentsschemasv0_0_43_openapi_warnings_item_data in _warnings or []:
            componentsschemasv0_0_43_openapi_warnings_item = V0043OpenapiWarning.from_dict(
                componentsschemasv0_0_43_openapi_warnings_item_data
            )

            warnings.append(componentsschemasv0_0_43_openapi_warnings_item)

        v0043_openapi_slurmdbd_config_resp = cls(
            clusters=clusters,
            tres=tres,
            accounts=accounts,
            users=users,
            qos=qos,
            wckeys=wckeys,
            associations=associations,
            instances=instances,
            meta=meta,
            errors=errors,
            warnings=warnings,
        )

        v0043_openapi_slurmdbd_config_resp.additional_properties = d
        return v0043_openapi_slurmdbd_config_resp

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
