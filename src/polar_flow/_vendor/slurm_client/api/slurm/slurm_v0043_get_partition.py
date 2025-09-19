from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.slurm_v0043_get_partition_flags import SlurmV0043GetPartitionFlags
from ...models.v0043_openapi_partition_resp import V0043OpenapiPartitionResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    partition_name: str,
    *,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetPartitionFlags] = UNSET,
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
        "url": f"/slurm/v0.0.43/partition/{partition_name}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiPartitionResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiPartitionResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiPartitionResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiPartitionResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    partition_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetPartitionFlags] = UNSET,
) -> Response[V0043OpenapiPartitionResp]:
    """get partition info

    Args:
        partition_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetPartitionFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiPartitionResp]
    """

    kwargs = _get_kwargs(
        partition_name=partition_name,
        update_time=update_time,
        flags=flags,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    partition_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetPartitionFlags] = UNSET,
) -> Optional[V0043OpenapiPartitionResp]:
    """get partition info

    Args:
        partition_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetPartitionFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiPartitionResp
    """

    return sync_detailed(
        partition_name=partition_name,
        client=client,
        update_time=update_time,
        flags=flags,
    ).parsed


async def asyncio_detailed(
    partition_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetPartitionFlags] = UNSET,
) -> Response[V0043OpenapiPartitionResp]:
    """get partition info

    Args:
        partition_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetPartitionFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiPartitionResp]
    """

    kwargs = _get_kwargs(
        partition_name=partition_name,
        update_time=update_time,
        flags=flags,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    partition_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043GetPartitionFlags] = UNSET,
) -> Optional[V0043OpenapiPartitionResp]:
    """get partition info

    Args:
        partition_name (str):
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043GetPartitionFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiPartitionResp
    """

    return (
        await asyncio_detailed(
            partition_name=partition_name,
            client=client,
            update_time=update_time,
            flags=flags,
        )
    ).parsed
