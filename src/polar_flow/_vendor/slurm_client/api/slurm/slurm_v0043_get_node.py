from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.slurm_v0043_get_node_flags import SlurmV0043GetNodeFlags
from ...models.v0043_openapi_nodes_resp import V0043OpenapiNodesResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    node_name: str,
    *,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetNodeFlags] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["update_time"] = update_time

    json_flags: Union[Unset, str] = UNSET
    if not isinstance(flags, Unset):
        json_flags = flags.value

    params["flags"] = json_flags

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/slurm/v0.0.43/node/{node_name}",
        "params": params,
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> V0043OpenapiNodesResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiNodesResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiNodesResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiNodesResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    node_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetNodeFlags] = UNSET,
) -> Response[V0043OpenapiNodesResp]:
    """get node info

    Args:
        node_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetNodeFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiNodesResp]
    """

    kwargs = _get_kwargs(
        node_name=node_name,
        update_time=update_time,
        flags=flags,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    node_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetNodeFlags] = UNSET,
) -> Optional[V0043OpenapiNodesResp]:
    """get node info

    Args:
        node_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetNodeFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiNodesResp
    """

    return sync_detailed(
        node_name=node_name,
        client=client,
        update_time=update_time,
        flags=flags,
    ).parsed


async def asyncio_detailed(
    node_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetNodeFlags] = UNSET,
) -> Response[V0043OpenapiNodesResp]:
    """get node info

    Args:
        node_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetNodeFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiNodesResp]
    """

    kwargs = _get_kwargs(
        node_name=node_name,
        update_time=update_time,
        flags=flags,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    node_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetNodeFlags] = UNSET,
) -> Optional[V0043OpenapiNodesResp]:
    """get node info

    Args:
        node_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetNodeFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiNodesResp
    """

    return (
        await asyncio_detailed(
            node_name=node_name,
            client=client,
            update_time=update_time,
            flags=flags,
        )
    ).parsed
