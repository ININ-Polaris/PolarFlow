from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_users_resp import V0043OpenapiUsersResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    name: str,
    *,
    with_deleted: Union[Unset, str] = UNSET,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_wckeys: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["with_deleted"] = with_deleted

    params["with_assocs"] = with_assocs

    params["with_coords"] = with_coords

    params["with_wckeys"] = with_wckeys

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/slurmdb/v0.0.43/user/{name}",
        "params": params,
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> V0043OpenapiUsersResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiUsersResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiUsersResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiUsersResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_wckeys: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiUsersResp]:
    """Get user info

    Args:
        name (str):
        with_deleted (Union[Unset, str]):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_wckeys (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiUsersResp]
    """

    kwargs = _get_kwargs(
        name=name,
        with_deleted=with_deleted,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_wckeys=with_wckeys,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_wckeys: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiUsersResp]:
    """Get user info

    Args:
        name (str):
        with_deleted (Union[Unset, str]):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_wckeys (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiUsersResp
    """

    return sync_detailed(
        name=name,
        client=client,
        with_deleted=with_deleted,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_wckeys=with_wckeys,
    ).parsed


async def asyncio_detailed(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_wckeys: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiUsersResp]:
    """Get user info

    Args:
        name (str):
        with_deleted (Union[Unset, str]):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_wckeys (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiUsersResp]
    """

    kwargs = _get_kwargs(
        name=name,
        with_deleted=with_deleted,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_wckeys=with_wckeys,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_deleted: Union[Unset, str] = UNSET,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_wckeys: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiUsersResp]:
    """Get user info

    Args:
        name (str):
        with_deleted (Union[Unset, str]):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_wckeys (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiUsersResp
    """

    return (
        await asyncio_detailed(
            name=name,
            client=client,
            with_deleted=with_deleted,
            with_assocs=with_assocs,
            with_coords=with_coords,
            with_wckeys=with_wckeys,
        )
    ).parsed
