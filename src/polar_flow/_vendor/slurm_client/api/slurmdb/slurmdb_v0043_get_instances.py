from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_instances_resp import V0043OpenapiInstancesResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    cluster: Union[Unset, str] = UNSET,
    extra: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    instance_id: Union[Unset, str] = UNSET,
    instance_type: Union[Unset, str] = UNSET,
    node_list: Union[Unset, str] = UNSET,
    time_end: Union[Unset, str] = UNSET,
    time_start: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["cluster"] = cluster

    params["extra"] = extra

    params["format"] = format_

    params["instance_id"] = instance_id

    params["instance_type"] = instance_type

    params["node_list"] = node_list

    params["time_end"] = time_end

    params["time_start"] = time_start

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/slurmdb/v0.0.43/instances/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiInstancesResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiInstancesResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiInstancesResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiInstancesResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    cluster: Union[Unset, str] = UNSET,
    extra: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    instance_id: Union[Unset, str] = UNSET,
    instance_type: Union[Unset, str] = UNSET,
    node_list: Union[Unset, str] = UNSET,
    time_end: Union[Unset, str] = UNSET,
    time_start: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiInstancesResp]:
    """Get instance list

    Args:
        cluster (Union[Unset, str]):
        extra (Union[Unset, str]):
        format_ (Union[Unset, str]):
        instance_id (Union[Unset, str]):
        instance_type (Union[Unset, str]):
        node_list (Union[Unset, str]):
        time_end (Union[Unset, str]):
        time_start (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiInstancesResp]
    """

    kwargs = _get_kwargs(
        cluster=cluster,
        extra=extra,
        format_=format_,
        instance_id=instance_id,
        instance_type=instance_type,
        node_list=node_list,
        time_end=time_end,
        time_start=time_start,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    cluster: Union[Unset, str] = UNSET,
    extra: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    instance_id: Union[Unset, str] = UNSET,
    instance_type: Union[Unset, str] = UNSET,
    node_list: Union[Unset, str] = UNSET,
    time_end: Union[Unset, str] = UNSET,
    time_start: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiInstancesResp]:
    """Get instance list

    Args:
        cluster (Union[Unset, str]):
        extra (Union[Unset, str]):
        format_ (Union[Unset, str]):
        instance_id (Union[Unset, str]):
        instance_type (Union[Unset, str]):
        node_list (Union[Unset, str]):
        time_end (Union[Unset, str]):
        time_start (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiInstancesResp
    """

    return sync_detailed(
        client=client,
        cluster=cluster,
        extra=extra,
        format_=format_,
        instance_id=instance_id,
        instance_type=instance_type,
        node_list=node_list,
        time_end=time_end,
        time_start=time_start,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    cluster: Union[Unset, str] = UNSET,
    extra: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    instance_id: Union[Unset, str] = UNSET,
    instance_type: Union[Unset, str] = UNSET,
    node_list: Union[Unset, str] = UNSET,
    time_end: Union[Unset, str] = UNSET,
    time_start: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiInstancesResp]:
    """Get instance list

    Args:
        cluster (Union[Unset, str]):
        extra (Union[Unset, str]):
        format_ (Union[Unset, str]):
        instance_id (Union[Unset, str]):
        instance_type (Union[Unset, str]):
        node_list (Union[Unset, str]):
        time_end (Union[Unset, str]):
        time_start (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiInstancesResp]
    """

    kwargs = _get_kwargs(
        cluster=cluster,
        extra=extra,
        format_=format_,
        instance_id=instance_id,
        instance_type=instance_type,
        node_list=node_list,
        time_end=time_end,
        time_start=time_start,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    cluster: Union[Unset, str] = UNSET,
    extra: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    instance_id: Union[Unset, str] = UNSET,
    instance_type: Union[Unset, str] = UNSET,
    node_list: Union[Unset, str] = UNSET,
    time_end: Union[Unset, str] = UNSET,
    time_start: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiInstancesResp]:
    """Get instance list

    Args:
        cluster (Union[Unset, str]):
        extra (Union[Unset, str]):
        format_ (Union[Unset, str]):
        instance_id (Union[Unset, str]):
        instance_type (Union[Unset, str]):
        node_list (Union[Unset, str]):
        time_end (Union[Unset, str]):
        time_start (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiInstancesResp
    """

    return (
        await asyncio_detailed(
            client=client,
            cluster=cluster,
            extra=extra,
            format_=format_,
            instance_id=instance_id,
            instance_type=instance_type,
            node_list=node_list,
            time_end=time_end,
            time_start=time_start,
        )
    ).parsed
