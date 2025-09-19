from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_assocs_resp import V0043OpenapiAssocsResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    account: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    default_qos: Union[Unset, str] = UNSET,
    include_deleted_associations: Union[Unset, str] = UNSET,
    include_usage: Union[Unset, str] = UNSET,
    filter_to_only_defaults: Union[Unset, str] = UNSET,
    include_the_raw_qos_or_delta_qos: Union[Unset, str] = UNSET,
    include_sub_acct_information: Union[Unset, str] = UNSET,
    exclude_parent_idname: Union[Unset, str] = UNSET,
    exclude_limits_from_parents: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    id: Union[Unset, str] = UNSET,
    parent_account: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    usage_end: Union[Unset, str] = UNSET,
    usage_start: Union[Unset, str] = UNSET,
    user: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["account"] = account

    params["cluster"] = cluster

    params["default_qos"] = default_qos

    params["Include deleted associations"] = include_deleted_associations

    params["Include usage"] = include_usage

    params["Filter to only defaults"] = filter_to_only_defaults

    params["Include the raw QOS or delta_qos"] = include_the_raw_qos_or_delta_qos

    params["Include sub acct information"] = include_sub_acct_information

    params["Exclude parent id/name"] = exclude_parent_idname

    params["Exclude limits from parents"] = exclude_limits_from_parents

    params["format"] = format_

    params["id"] = id

    params["parent_account"] = parent_account

    params["partition"] = partition

    params["qos"] = qos

    params["usage_end"] = usage_end

    params["usage_start"] = usage_start

    params["user"] = user

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/slurmdb/v0.0.43/associations/",
        "params": params,
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> V0043OpenapiAssocsResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiAssocsResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiAssocsResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiAssocsResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    default_qos: Union[Unset, str] = UNSET,
    include_deleted_associations: Union[Unset, str] = UNSET,
    include_usage: Union[Unset, str] = UNSET,
    filter_to_only_defaults: Union[Unset, str] = UNSET,
    include_the_raw_qos_or_delta_qos: Union[Unset, str] = UNSET,
    include_sub_acct_information: Union[Unset, str] = UNSET,
    exclude_parent_idname: Union[Unset, str] = UNSET,
    exclude_limits_from_parents: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    id: Union[Unset, str] = UNSET,
    parent_account: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    usage_end: Union[Unset, str] = UNSET,
    usage_start: Union[Unset, str] = UNSET,
    user: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiAssocsResp]:
    """Get association list

    Args:
        account (Union[Unset, str]):
        cluster (Union[Unset, str]):
        default_qos (Union[Unset, str]):
        include_deleted_associations (Union[Unset, str]):
        include_usage (Union[Unset, str]):
        filter_to_only_defaults (Union[Unset, str]):
        include_the_raw_qos_or_delta_qos (Union[Unset, str]):
        include_sub_acct_information (Union[Unset, str]):
        exclude_parent_idname (Union[Unset, str]):
        exclude_limits_from_parents (Union[Unset, str]):
        format_ (Union[Unset, str]):
        id (Union[Unset, str]):
        parent_account (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        usage_end (Union[Unset, str]):
        usage_start (Union[Unset, str]):
        user (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAssocsResp]
    """

    kwargs = _get_kwargs(
        account=account,
        cluster=cluster,
        default_qos=default_qos,
        include_deleted_associations=include_deleted_associations,
        include_usage=include_usage,
        filter_to_only_defaults=filter_to_only_defaults,
        include_the_raw_qos_or_delta_qos=include_the_raw_qos_or_delta_qos,
        include_sub_acct_information=include_sub_acct_information,
        exclude_parent_idname=exclude_parent_idname,
        exclude_limits_from_parents=exclude_limits_from_parents,
        format_=format_,
        id=id,
        parent_account=parent_account,
        partition=partition,
        qos=qos,
        usage_end=usage_end,
        usage_start=usage_start,
        user=user,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    default_qos: Union[Unset, str] = UNSET,
    include_deleted_associations: Union[Unset, str] = UNSET,
    include_usage: Union[Unset, str] = UNSET,
    filter_to_only_defaults: Union[Unset, str] = UNSET,
    include_the_raw_qos_or_delta_qos: Union[Unset, str] = UNSET,
    include_sub_acct_information: Union[Unset, str] = UNSET,
    exclude_parent_idname: Union[Unset, str] = UNSET,
    exclude_limits_from_parents: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    id: Union[Unset, str] = UNSET,
    parent_account: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    usage_end: Union[Unset, str] = UNSET,
    usage_start: Union[Unset, str] = UNSET,
    user: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiAssocsResp]:
    """Get association list

    Args:
        account (Union[Unset, str]):
        cluster (Union[Unset, str]):
        default_qos (Union[Unset, str]):
        include_deleted_associations (Union[Unset, str]):
        include_usage (Union[Unset, str]):
        filter_to_only_defaults (Union[Unset, str]):
        include_the_raw_qos_or_delta_qos (Union[Unset, str]):
        include_sub_acct_information (Union[Unset, str]):
        exclude_parent_idname (Union[Unset, str]):
        exclude_limits_from_parents (Union[Unset, str]):
        format_ (Union[Unset, str]):
        id (Union[Unset, str]):
        parent_account (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        usage_end (Union[Unset, str]):
        usage_start (Union[Unset, str]):
        user (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAssocsResp
    """

    return sync_detailed(
        client=client,
        account=account,
        cluster=cluster,
        default_qos=default_qos,
        include_deleted_associations=include_deleted_associations,
        include_usage=include_usage,
        filter_to_only_defaults=filter_to_only_defaults,
        include_the_raw_qos_or_delta_qos=include_the_raw_qos_or_delta_qos,
        include_sub_acct_information=include_sub_acct_information,
        exclude_parent_idname=exclude_parent_idname,
        exclude_limits_from_parents=exclude_limits_from_parents,
        format_=format_,
        id=id,
        parent_account=parent_account,
        partition=partition,
        qos=qos,
        usage_end=usage_end,
        usage_start=usage_start,
        user=user,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    default_qos: Union[Unset, str] = UNSET,
    include_deleted_associations: Union[Unset, str] = UNSET,
    include_usage: Union[Unset, str] = UNSET,
    filter_to_only_defaults: Union[Unset, str] = UNSET,
    include_the_raw_qos_or_delta_qos: Union[Unset, str] = UNSET,
    include_sub_acct_information: Union[Unset, str] = UNSET,
    exclude_parent_idname: Union[Unset, str] = UNSET,
    exclude_limits_from_parents: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    id: Union[Unset, str] = UNSET,
    parent_account: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    usage_end: Union[Unset, str] = UNSET,
    usage_start: Union[Unset, str] = UNSET,
    user: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiAssocsResp]:
    """Get association list

    Args:
        account (Union[Unset, str]):
        cluster (Union[Unset, str]):
        default_qos (Union[Unset, str]):
        include_deleted_associations (Union[Unset, str]):
        include_usage (Union[Unset, str]):
        filter_to_only_defaults (Union[Unset, str]):
        include_the_raw_qos_or_delta_qos (Union[Unset, str]):
        include_sub_acct_information (Union[Unset, str]):
        exclude_parent_idname (Union[Unset, str]):
        exclude_limits_from_parents (Union[Unset, str]):
        format_ (Union[Unset, str]):
        id (Union[Unset, str]):
        parent_account (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        usage_end (Union[Unset, str]):
        usage_start (Union[Unset, str]):
        user (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAssocsResp]
    """

    kwargs = _get_kwargs(
        account=account,
        cluster=cluster,
        default_qos=default_qos,
        include_deleted_associations=include_deleted_associations,
        include_usage=include_usage,
        filter_to_only_defaults=filter_to_only_defaults,
        include_the_raw_qos_or_delta_qos=include_the_raw_qos_or_delta_qos,
        include_sub_acct_information=include_sub_acct_information,
        exclude_parent_idname=exclude_parent_idname,
        exclude_limits_from_parents=exclude_limits_from_parents,
        format_=format_,
        id=id,
        parent_account=parent_account,
        partition=partition,
        qos=qos,
        usage_end=usage_end,
        usage_start=usage_start,
        user=user,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    default_qos: Union[Unset, str] = UNSET,
    include_deleted_associations: Union[Unset, str] = UNSET,
    include_usage: Union[Unset, str] = UNSET,
    filter_to_only_defaults: Union[Unset, str] = UNSET,
    include_the_raw_qos_or_delta_qos: Union[Unset, str] = UNSET,
    include_sub_acct_information: Union[Unset, str] = UNSET,
    exclude_parent_idname: Union[Unset, str] = UNSET,
    exclude_limits_from_parents: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    id: Union[Unset, str] = UNSET,
    parent_account: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    usage_end: Union[Unset, str] = UNSET,
    usage_start: Union[Unset, str] = UNSET,
    user: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiAssocsResp]:
    """Get association list

    Args:
        account (Union[Unset, str]):
        cluster (Union[Unset, str]):
        default_qos (Union[Unset, str]):
        include_deleted_associations (Union[Unset, str]):
        include_usage (Union[Unset, str]):
        filter_to_only_defaults (Union[Unset, str]):
        include_the_raw_qos_or_delta_qos (Union[Unset, str]):
        include_sub_acct_information (Union[Unset, str]):
        exclude_parent_idname (Union[Unset, str]):
        exclude_limits_from_parents (Union[Unset, str]):
        format_ (Union[Unset, str]):
        id (Union[Unset, str]):
        parent_account (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        usage_end (Union[Unset, str]):
        usage_start (Union[Unset, str]):
        user (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAssocsResp
    """

    return (
        await asyncio_detailed(
            client=client,
            account=account,
            cluster=cluster,
            default_qos=default_qos,
            include_deleted_associations=include_deleted_associations,
            include_usage=include_usage,
            filter_to_only_defaults=filter_to_only_defaults,
            include_the_raw_qos_or_delta_qos=include_the_raw_qos_or_delta_qos,
            include_sub_acct_information=include_sub_acct_information,
            exclude_parent_idname=exclude_parent_idname,
            exclude_limits_from_parents=exclude_limits_from_parents,
            format_=format_,
            id=id,
            parent_account=parent_account,
            partition=partition,
            qos=qos,
            usage_end=usage_end,
            usage_start=usage_start,
            user=user,
        )
    ).parsed
