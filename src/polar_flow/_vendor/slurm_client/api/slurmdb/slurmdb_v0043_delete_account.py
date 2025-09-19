from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_accounts_removed_resp import V0043OpenapiAccountsRemovedResp
from ...types import Response


def _get_kwargs(
    account_name: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/slurmdb/v0.0.43/account/{account_name}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiAccountsRemovedResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiAccountsRemovedResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiAccountsRemovedResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiAccountsRemovedResp]:
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
) -> Response[V0043OpenapiAccountsRemovedResp]:
    """Delete account

    Args:
        account_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAccountsRemovedResp]
    """

    kwargs = _get_kwargs(
        account_name=account_name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[V0043OpenapiAccountsRemovedResp]:
    """Delete account

    Args:
        account_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAccountsRemovedResp
    """

    return sync_detailed(
        account_name=account_name,
        client=client,
    ).parsed


async def asyncio_detailed(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[V0043OpenapiAccountsRemovedResp]:
    """Delete account

    Args:
        account_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiAccountsRemovedResp]
    """

    kwargs = _get_kwargs(
        account_name=account_name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    account_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[V0043OpenapiAccountsRemovedResp]:
    """Delete account

    Args:
        account_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiAccountsRemovedResp
    """

    return (
        await asyncio_detailed(
            account_name=account_name,
            client=client,
        )
    ).parsed
