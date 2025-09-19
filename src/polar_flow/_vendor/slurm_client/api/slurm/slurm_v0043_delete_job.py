from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.slurm_v0043_delete_job_flags import SlurmV0043DeleteJobFlags
from ...models.v0043_openapi_kill_job_resp import V0043OpenapiKillJobResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    job_id: str,
    *,
    signal: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043DeleteJobFlags] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["signal"] = signal

    json_flags: Union[Unset, str] = UNSET
    if not isinstance(flags, Unset):
        json_flags = flags.value

    params["flags"] = json_flags

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/slurm/v0.0.43/job/{job_id}",
        "params": params,
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> V0043OpenapiKillJobResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiKillJobResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiKillJobResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiKillJobResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    job_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    signal: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043DeleteJobFlags] = UNSET,
) -> Response[V0043OpenapiKillJobResp]:
    """cancel or signal job

    Args:
        job_id (str):
        signal (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043DeleteJobFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiKillJobResp]
    """

    kwargs = _get_kwargs(
        job_id=job_id,
        signal=signal,
        flags=flags,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    job_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    signal: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043DeleteJobFlags] = UNSET,
) -> Optional[V0043OpenapiKillJobResp]:
    """cancel or signal job

    Args:
        job_id (str):
        signal (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043DeleteJobFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiKillJobResp
    """

    return sync_detailed(
        job_id=job_id,
        client=client,
        signal=signal,
        flags=flags,
    ).parsed


async def asyncio_detailed(
    job_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    signal: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043DeleteJobFlags] = UNSET,
) -> Response[V0043OpenapiKillJobResp]:
    """cancel or signal job

    Args:
        job_id (str):
        signal (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043DeleteJobFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiKillJobResp]
    """

    kwargs = _get_kwargs(
        job_id=job_id,
        signal=signal,
        flags=flags,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    job_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    signal: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmV0043DeleteJobFlags] = UNSET,
) -> Optional[V0043OpenapiKillJobResp]:
    """cancel or signal job

    Args:
        job_id (str):
        signal (Union[Unset, str]):
        flags (Union[Unset, SlurmV0043DeleteJobFlags]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiKillJobResp
    """

    return (
        await asyncio_detailed(
            job_id=job_id,
            client=client,
            signal=signal,
            flags=flags,
        )
    ).parsed
