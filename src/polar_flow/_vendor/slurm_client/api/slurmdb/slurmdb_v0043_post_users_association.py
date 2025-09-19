from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.slurmdb_v0043_post_users_association_flags import SlurmdbV0043PostUsersAssociationFlags
from ...models.v0043_openapi_users_add_cond_resp import V0043OpenapiUsersAddCondResp
from ...models.v0043_openapi_users_add_cond_resp_str import V0043OpenapiUsersAddCondRespStr
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: V0043OpenapiUsersAddCondResp,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmdbV0043PostUsersAssociationFlags] = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["update_time"] = update_time

    json_flags: Union[Unset, str] = UNSET
    if not isinstance(flags, Unset):
        json_flags = flags.value

    params["flags"] = json_flags

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/slurmdb/v0.0.43/users_association/",
        "params": params,
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiUsersAddCondRespStr:
    if response.status_code == 200:
        response_200 = V0043OpenapiUsersAddCondRespStr.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiUsersAddCondRespStr.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiUsersAddCondRespStr]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: V0043OpenapiUsersAddCondResp,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmdbV0043PostUsersAssociationFlags] = UNSET,
) -> Response[V0043OpenapiUsersAddCondRespStr]:
    """Add users with conditional association

    Args:
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmdbV0043PostUsersAssociationFlags]):
        body (V0043OpenapiUsersAddCondResp):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiUsersAddCondRespStr]
    """

    kwargs = _get_kwargs(
        body=body,
        update_time=update_time,
        flags=flags,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: V0043OpenapiUsersAddCondResp,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmdbV0043PostUsersAssociationFlags] = UNSET,
) -> Optional[V0043OpenapiUsersAddCondRespStr]:
    """Add users with conditional association

    Args:
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmdbV0043PostUsersAssociationFlags]):
        body (V0043OpenapiUsersAddCondResp):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiUsersAddCondRespStr
    """

    return sync_detailed(
        client=client,
        body=body,
        update_time=update_time,
        flags=flags,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: V0043OpenapiUsersAddCondResp,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmdbV0043PostUsersAssociationFlags] = UNSET,
) -> Response[V0043OpenapiUsersAddCondRespStr]:
    """Add users with conditional association

    Args:
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmdbV0043PostUsersAssociationFlags]):
        body (V0043OpenapiUsersAddCondResp):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiUsersAddCondRespStr]
    """

    kwargs = _get_kwargs(
        body=body,
        update_time=update_time,
        flags=flags,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: V0043OpenapiUsersAddCondResp,
    update_time: Union[Unset, str] = UNSET,
    flags: Union[Unset, SlurmdbV0043PostUsersAssociationFlags] = UNSET,
) -> Optional[V0043OpenapiUsersAddCondRespStr]:
    """Add users with conditional association

    Args:
        update_time (Union[Unset, str]):
        flags (Union[Unset, SlurmdbV0043PostUsersAssociationFlags]):
        body (V0043OpenapiUsersAddCondResp):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiUsersAddCondRespStr
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            update_time=update_time,
            flags=flags,
        )
    ).parsed
