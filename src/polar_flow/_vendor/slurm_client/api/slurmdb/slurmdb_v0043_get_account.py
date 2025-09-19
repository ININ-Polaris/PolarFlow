from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_accounts_resp import V0043OpenapiAccountsResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    account_name: str,
    *,
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_deleted: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["with_assocs"] = with_assocs

    params["with_coords"] = with_coords

    params["with_deleted"] = with_deleted

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/slurmdb/v0.0.43/account/{account_name}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiAccountsResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiAccountsResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiAccountsResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiAccountsResp]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_deleted: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiAccountsResp]:
    """Get account info

    Args:
        account_name (str):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAccountsResp]
    """

    kwargs = _get_kwargs(
        account_name=account_name,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_deleted=with_deleted,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_deleted: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiAccountsResp]:
    """Get account info

    Args:
        account_name (str):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAccountsResp
    """

    return sync_detailed(
        account_name=account_name,
        client=client,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_deleted=with_deleted,
    ).parsed


async def asyncio_detailed(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_deleted: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiAccountsResp]:
    """Get account info

    Args:
        account_name (str):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAccountsResp]
    """

    kwargs = _get_kwargs(
        account_name=account_name,
        with_assocs=with_assocs,
        with_coords=with_coords,
        with_deleted=with_deleted,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    with_assocs: Union[Unset, str] = UNSET,
    with_coords: Union[Unset, str] = UNSET,
    with_deleted: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiAccountsResp]:
    """Get account info

    Args:
        account_name (str):
        with_assocs (Union[Unset, str]):
        with_coords (Union[Unset, str]):
        with_deleted (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAccountsResp
    """

    return (
        await asyncio_detailed(
            account_name=account_name,
            client=client,
            with_assocs=with_assocs,
            with_coords=with_coords,
            with_deleted=with_deleted,
        )
    ).parsed
