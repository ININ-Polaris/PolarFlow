from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_slurmdbd_qos_resp import V0043OpenapiSlurmdbdQosResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    qos: str,
    *,
    with_deleted: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["with_deleted"] = with_deleted

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/slurmdb/v0.0.43/qos/{qos}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiSlurmdbdQosResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiSlurmdbdQosResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiSlurmdbdQosResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiSlurmdbdQosResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    qos: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiSlurmdbdQosResp]:
    """Get QOS info

    Args:
        qos (str):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiSlurmdbdQosResp]
    """

    kwargs = _get_kwargs(
        qos=qos,
        with_deleted=with_deleted,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    qos: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiSlurmdbdQosResp]:
    """Get QOS info

    Args:
        qos (str):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiSlurmdbdQosResp
    """

    return sync_detailed(
        qos=qos,
        client=client,
        with_deleted=with_deleted,
    ).parsed


async def asyncio_detailed(
    qos: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiSlurmdbdQosResp]:
    """Get QOS info

    Args:
        qos (str):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiSlurmdbdQosResp]
    """

    kwargs = _get_kwargs(
        qos=qos,
        with_deleted=with_deleted,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    qos: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiSlurmdbdQosResp]:
    """Get QOS info

    Args:
        qos (str):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiSlurmdbdQosResp
    """

    return (
        await asyncio_detailed(
            qos=qos,
            client=client,
            with_deleted=with_deleted,
        )
    ).parsed
