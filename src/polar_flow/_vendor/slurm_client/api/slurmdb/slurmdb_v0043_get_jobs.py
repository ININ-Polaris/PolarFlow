from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ...client import AuthenticatedClient, Client
from ...models.v0043_openapi_slurmdbd_jobs_resp import V0043OpenapiSlurmdbdJobsResp
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    account: Union[Unset, str] = UNSET,
    association: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    constraints: Union[Unset, str] = UNSET,
    scheduler_unset: Union[Unset, str] = UNSET,
    scheduled_on_submit: Union[Unset, str] = UNSET,
    scheduled_by_main: Union[Unset, str] = UNSET,
    scheduled_by_backfill: Union[Unset, str] = UNSET,
    job_started: Union[Unset, str] = UNSET,
    exit_code: Union[Unset, str] = UNSET,
    show_duplicates: Union[Unset, str] = UNSET,
    skip_steps: Union[Unset, str] = UNSET,
    disable_truncate_usage_time: Union[Unset, str] = UNSET,
    whole_hetjob: Union[Unset, str] = UNSET,
    disable_whole_hetjob: Union[Unset, str] = UNSET,
    disable_wait_for_result: Union[Unset, str] = UNSET,
    usage_time_as_submit_time: Union[Unset, str] = UNSET,
    show_batch_script: Union[Unset, str] = UNSET,
    show_job_environment: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    groups: Union[Unset, str] = UNSET,
    job_name: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    reason: Union[Unset, str] = UNSET,
    reservation: Union[Unset, str] = UNSET,
    reservation_id: Union[Unset, str] = UNSET,
    state: Union[Unset, str] = UNSET,
    step: Union[Unset, str] = UNSET,
    end_time: Union[Unset, str] = UNSET,
    start_time: Union[Unset, str] = UNSET,
    node: Union[Unset, str] = UNSET,
    users: Union[Unset, str] = UNSET,
    wckey: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["account"] = account

    params["association"] = association

    params["cluster"] = cluster

    params["constraints"] = constraints

    params["scheduler_unset"] = scheduler_unset

    params["scheduled_on_submit"] = scheduled_on_submit

    params["scheduled_by_main"] = scheduled_by_main

    params["scheduled_by_backfill"] = scheduled_by_backfill

    params["job_started"] = job_started

    params["exit_code"] = exit_code

    params["show_duplicates"] = show_duplicates

    params["skip_steps"] = skip_steps

    params["disable_truncate_usage_time"] = disable_truncate_usage_time

    params["whole_hetjob"] = whole_hetjob

    params["disable_whole_hetjob"] = disable_whole_hetjob

    params["disable_wait_for_result"] = disable_wait_for_result

    params["usage_time_as_submit_time"] = usage_time_as_submit_time

    params["show_batch_script"] = show_batch_script

    params["show_job_environment"] = show_job_environment

    params["format"] = format_

    params["groups"] = groups

    params["job_name"] = job_name

    params["partition"] = partition

    params["qos"] = qos

    params["reason"] = reason

    params["reservation"] = reservation

    params["reservation_id"] = reservation_id

    params["state"] = state

    params["step"] = step

    params["end_time"] = end_time

    params["start_time"] = start_time

    params["node"] = node

    params["users"] = users

    params["wckey"] = wckey

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/slurmdb/v0.0.43/jobs/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> V0043OpenapiSlurmdbdJobsResp:
    if response.status_code == 200:
        response_200 = V0043OpenapiSlurmdbdJobsResp.from_dict(response.json())

        return response_200

    response_default = V0043OpenapiSlurmdbdJobsResp.from_dict(response.json())

    return response_default


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[V0043OpenapiSlurmdbdJobsResp]:
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
    association: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    constraints: Union[Unset, str] = UNSET,
    scheduler_unset: Union[Unset, str] = UNSET,
    scheduled_on_submit: Union[Unset, str] = UNSET,
    scheduled_by_main: Union[Unset, str] = UNSET,
    scheduled_by_backfill: Union[Unset, str] = UNSET,
    job_started: Union[Unset, str] = UNSET,
    exit_code: Union[Unset, str] = UNSET,
    show_duplicates: Union[Unset, str] = UNSET,
    skip_steps: Union[Unset, str] = UNSET,
    disable_truncate_usage_time: Union[Unset, str] = UNSET,
    whole_hetjob: Union[Unset, str] = UNSET,
    disable_whole_hetjob: Union[Unset, str] = UNSET,
    disable_wait_for_result: Union[Unset, str] = UNSET,
    usage_time_as_submit_time: Union[Unset, str] = UNSET,
    show_batch_script: Union[Unset, str] = UNSET,
    show_job_environment: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    groups: Union[Unset, str] = UNSET,
    job_name: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    reason: Union[Unset, str] = UNSET,
    reservation: Union[Unset, str] = UNSET,
    reservation_id: Union[Unset, str] = UNSET,
    state: Union[Unset, str] = UNSET,
    step: Union[Unset, str] = UNSET,
    end_time: Union[Unset, str] = UNSET,
    start_time: Union[Unset, str] = UNSET,
    node: Union[Unset, str] = UNSET,
    users: Union[Unset, str] = UNSET,
    wckey: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiSlurmdbdJobsResp]:
    """Get job list

    Args:
        account (Union[Unset, str]):
        association (Union[Unset, str]):
        cluster (Union[Unset, str]):
        constraints (Union[Unset, str]):
        scheduler_unset (Union[Unset, str]):
        scheduled_on_submit (Union[Unset, str]):
        scheduled_by_main (Union[Unset, str]):
        scheduled_by_backfill (Union[Unset, str]):
        job_started (Union[Unset, str]):
        exit_code (Union[Unset, str]):
        show_duplicates (Union[Unset, str]):
        skip_steps (Union[Unset, str]):
        disable_truncate_usage_time (Union[Unset, str]):
        whole_hetjob (Union[Unset, str]):
        disable_whole_hetjob (Union[Unset, str]):
        disable_wait_for_result (Union[Unset, str]):
        usage_time_as_submit_time (Union[Unset, str]):
        show_batch_script (Union[Unset, str]):
        show_job_environment (Union[Unset, str]):
        format_ (Union[Unset, str]):
        groups (Union[Unset, str]):
        job_name (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        reason (Union[Unset, str]):
        reservation (Union[Unset, str]):
        reservation_id (Union[Unset, str]):
        state (Union[Unset, str]):
        step (Union[Unset, str]):
        end_time (Union[Unset, str]):
        start_time (Union[Unset, str]):
        node (Union[Unset, str]):
        users (Union[Unset, str]):
        wckey (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiSlurmdbdJobsResp]
    """

    kwargs = _get_kwargs(
        account=account,
        association=association,
        cluster=cluster,
        constraints=constraints,
        scheduler_unset=scheduler_unset,
        scheduled_on_submit=scheduled_on_submit,
        scheduled_by_main=scheduled_by_main,
        scheduled_by_backfill=scheduled_by_backfill,
        job_started=job_started,
        exit_code=exit_code,
        show_duplicates=show_duplicates,
        skip_steps=skip_steps,
        disable_truncate_usage_time=disable_truncate_usage_time,
        whole_hetjob=whole_hetjob,
        disable_whole_hetjob=disable_whole_hetjob,
        disable_wait_for_result=disable_wait_for_result,
        usage_time_as_submit_time=usage_time_as_submit_time,
        show_batch_script=show_batch_script,
        show_job_environment=show_job_environment,
        format_=format_,
        groups=groups,
        job_name=job_name,
        partition=partition,
        qos=qos,
        reason=reason,
        reservation=reservation,
        reservation_id=reservation_id,
        state=state,
        step=step,
        end_time=end_time,
        start_time=start_time,
        node=node,
        users=users,
        wckey=wckey,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    association: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    constraints: Union[Unset, str] = UNSET,
    scheduler_unset: Union[Unset, str] = UNSET,
    scheduled_on_submit: Union[Unset, str] = UNSET,
    scheduled_by_main: Union[Unset, str] = UNSET,
    scheduled_by_backfill: Union[Unset, str] = UNSET,
    job_started: Union[Unset, str] = UNSET,
    exit_code: Union[Unset, str] = UNSET,
    show_duplicates: Union[Unset, str] = UNSET,
    skip_steps: Union[Unset, str] = UNSET,
    disable_truncate_usage_time: Union[Unset, str] = UNSET,
    whole_hetjob: Union[Unset, str] = UNSET,
    disable_whole_hetjob: Union[Unset, str] = UNSET,
    disable_wait_for_result: Union[Unset, str] = UNSET,
    usage_time_as_submit_time: Union[Unset, str] = UNSET,
    show_batch_script: Union[Unset, str] = UNSET,
    show_job_environment: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    groups: Union[Unset, str] = UNSET,
    job_name: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    reason: Union[Unset, str] = UNSET,
    reservation: Union[Unset, str] = UNSET,
    reservation_id: Union[Unset, str] = UNSET,
    state: Union[Unset, str] = UNSET,
    step: Union[Unset, str] = UNSET,
    end_time: Union[Unset, str] = UNSET,
    start_time: Union[Unset, str] = UNSET,
    node: Union[Unset, str] = UNSET,
    users: Union[Unset, str] = UNSET,
    wckey: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiSlurmdbdJobsResp]:
    """Get job list

    Args:
        account (Union[Unset, str]):
        association (Union[Unset, str]):
        cluster (Union[Unset, str]):
        constraints (Union[Unset, str]):
        scheduler_unset (Union[Unset, str]):
        scheduled_on_submit (Union[Unset, str]):
        scheduled_by_main (Union[Unset, str]):
        scheduled_by_backfill (Union[Unset, str]):
        job_started (Union[Unset, str]):
        exit_code (Union[Unset, str]):
        show_duplicates (Union[Unset, str]):
        skip_steps (Union[Unset, str]):
        disable_truncate_usage_time (Union[Unset, str]):
        whole_hetjob (Union[Unset, str]):
        disable_whole_hetjob (Union[Unset, str]):
        disable_wait_for_result (Union[Unset, str]):
        usage_time_as_submit_time (Union[Unset, str]):
        show_batch_script (Union[Unset, str]):
        show_job_environment (Union[Unset, str]):
        format_ (Union[Unset, str]):
        groups (Union[Unset, str]):
        job_name (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        reason (Union[Unset, str]):
        reservation (Union[Unset, str]):
        reservation_id (Union[Unset, str]):
        state (Union[Unset, str]):
        step (Union[Unset, str]):
        end_time (Union[Unset, str]):
        start_time (Union[Unset, str]):
        node (Union[Unset, str]):
        users (Union[Unset, str]):
        wckey (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiSlurmdbdJobsResp
    """

    return sync_detailed(
        client=client,
        account=account,
        association=association,
        cluster=cluster,
        constraints=constraints,
        scheduler_unset=scheduler_unset,
        scheduled_on_submit=scheduled_on_submit,
        scheduled_by_main=scheduled_by_main,
        scheduled_by_backfill=scheduled_by_backfill,
        job_started=job_started,
        exit_code=exit_code,
        show_duplicates=show_duplicates,
        skip_steps=skip_steps,
        disable_truncate_usage_time=disable_truncate_usage_time,
        whole_hetjob=whole_hetjob,
        disable_whole_hetjob=disable_whole_hetjob,
        disable_wait_for_result=disable_wait_for_result,
        usage_time_as_submit_time=usage_time_as_submit_time,
        show_batch_script=show_batch_script,
        show_job_environment=show_job_environment,
        format_=format_,
        groups=groups,
        job_name=job_name,
        partition=partition,
        qos=qos,
        reason=reason,
        reservation=reservation,
        reservation_id=reservation_id,
        state=state,
        step=step,
        end_time=end_time,
        start_time=start_time,
        node=node,
        users=users,
        wckey=wckey,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    association: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    constraints: Union[Unset, str] = UNSET,
    scheduler_unset: Union[Unset, str] = UNSET,
    scheduled_on_submit: Union[Unset, str] = UNSET,
    scheduled_by_main: Union[Unset, str] = UNSET,
    scheduled_by_backfill: Union[Unset, str] = UNSET,
    job_started: Union[Unset, str] = UNSET,
    exit_code: Union[Unset, str] = UNSET,
    show_duplicates: Union[Unset, str] = UNSET,
    skip_steps: Union[Unset, str] = UNSET,
    disable_truncate_usage_time: Union[Unset, str] = UNSET,
    whole_hetjob: Union[Unset, str] = UNSET,
    disable_whole_hetjob: Union[Unset, str] = UNSET,
    disable_wait_for_result: Union[Unset, str] = UNSET,
    usage_time_as_submit_time: Union[Unset, str] = UNSET,
    show_batch_script: Union[Unset, str] = UNSET,
    show_job_environment: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    groups: Union[Unset, str] = UNSET,
    job_name: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    reason: Union[Unset, str] = UNSET,
    reservation: Union[Unset, str] = UNSET,
    reservation_id: Union[Unset, str] = UNSET,
    state: Union[Unset, str] = UNSET,
    step: Union[Unset, str] = UNSET,
    end_time: Union[Unset, str] = UNSET,
    start_time: Union[Unset, str] = UNSET,
    node: Union[Unset, str] = UNSET,
    users: Union[Unset, str] = UNSET,
    wckey: Union[Unset, str] = UNSET,
) -> Response[V0043OpenapiSlurmdbdJobsResp]:
    """Get job list

    Args:
        account (Union[Unset, str]):
        association (Union[Unset, str]):
        cluster (Union[Unset, str]):
        constraints (Union[Unset, str]):
        scheduler_unset (Union[Unset, str]):
        scheduled_on_submit (Union[Unset, str]):
        scheduled_by_main (Union[Unset, str]):
        scheduled_by_backfill (Union[Unset, str]):
        job_started (Union[Unset, str]):
        exit_code (Union[Unset, str]):
        show_duplicates (Union[Unset, str]):
        skip_steps (Union[Unset, str]):
        disable_truncate_usage_time (Union[Unset, str]):
        whole_hetjob (Union[Unset, str]):
        disable_whole_hetjob (Union[Unset, str]):
        disable_wait_for_result (Union[Unset, str]):
        usage_time_as_submit_time (Union[Unset, str]):
        show_batch_script (Union[Unset, str]):
        show_job_environment (Union[Unset, str]):
        format_ (Union[Unset, str]):
        groups (Union[Unset, str]):
        job_name (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        reason (Union[Unset, str]):
        reservation (Union[Unset, str]):
        reservation_id (Union[Unset, str]):
        state (Union[Unset, str]):
        step (Union[Unset, str]):
        end_time (Union[Unset, str]):
        start_time (Union[Unset, str]):
        node (Union[Unset, str]):
        users (Union[Unset, str]):
        wckey (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[V0043OpenapiSlurmdbdJobsResp]
    """

    kwargs = _get_kwargs(
        account=account,
        association=association,
        cluster=cluster,
        constraints=constraints,
        scheduler_unset=scheduler_unset,
        scheduled_on_submit=scheduled_on_submit,
        scheduled_by_main=scheduled_by_main,
        scheduled_by_backfill=scheduled_by_backfill,
        job_started=job_started,
        exit_code=exit_code,
        show_duplicates=show_duplicates,
        skip_steps=skip_steps,
        disable_truncate_usage_time=disable_truncate_usage_time,
        whole_hetjob=whole_hetjob,
        disable_whole_hetjob=disable_whole_hetjob,
        disable_wait_for_result=disable_wait_for_result,
        usage_time_as_submit_time=usage_time_as_submit_time,
        show_batch_script=show_batch_script,
        show_job_environment=show_job_environment,
        format_=format_,
        groups=groups,
        job_name=job_name,
        partition=partition,
        qos=qos,
        reason=reason,
        reservation=reservation,
        reservation_id=reservation_id,
        state=state,
        step=step,
        end_time=end_time,
        start_time=start_time,
        node=node,
        users=users,
        wckey=wckey,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    account: Union[Unset, str] = UNSET,
    association: Union[Unset, str] = UNSET,
    cluster: Union[Unset, str] = UNSET,
    constraints: Union[Unset, str] = UNSET,
    scheduler_unset: Union[Unset, str] = UNSET,
    scheduled_on_submit: Union[Unset, str] = UNSET,
    scheduled_by_main: Union[Unset, str] = UNSET,
    scheduled_by_backfill: Union[Unset, str] = UNSET,
    job_started: Union[Unset, str] = UNSET,
    exit_code: Union[Unset, str] = UNSET,
    show_duplicates: Union[Unset, str] = UNSET,
    skip_steps: Union[Unset, str] = UNSET,
    disable_truncate_usage_time: Union[Unset, str] = UNSET,
    whole_hetjob: Union[Unset, str] = UNSET,
    disable_whole_hetjob: Union[Unset, str] = UNSET,
    disable_wait_for_result: Union[Unset, str] = UNSET,
    usage_time_as_submit_time: Union[Unset, str] = UNSET,
    show_batch_script: Union[Unset, str] = UNSET,
    show_job_environment: Union[Unset, str] = UNSET,
    format_: Union[Unset, str] = UNSET,
    groups: Union[Unset, str] = UNSET,
    job_name: Union[Unset, str] = UNSET,
    partition: Union[Unset, str] = UNSET,
    qos: Union[Unset, str] = UNSET,
    reason: Union[Unset, str] = UNSET,
    reservation: Union[Unset, str] = UNSET,
    reservation_id: Union[Unset, str] = UNSET,
    state: Union[Unset, str] = UNSET,
    step: Union[Unset, str] = UNSET,
    end_time: Union[Unset, str] = UNSET,
    start_time: Union[Unset, str] = UNSET,
    node: Union[Unset, str] = UNSET,
    users: Union[Unset, str] = UNSET,
    wckey: Union[Unset, str] = UNSET,
) -> Optional[V0043OpenapiSlurmdbdJobsResp]:
    """Get job list

    Args:
        account (Union[Unset, str]):
        association (Union[Unset, str]):
        cluster (Union[Unset, str]):
        constraints (Union[Unset, str]):
        scheduler_unset (Union[Unset, str]):
        scheduled_on_submit (Union[Unset, str]):
        scheduled_by_main (Union[Unset, str]):
        scheduled_by_backfill (Union[Unset, str]):
        job_started (Union[Unset, str]):
        exit_code (Union[Unset, str]):
        show_duplicates (Union[Unset, str]):
        skip_steps (Union[Unset, str]):
        disable_truncate_usage_time (Union[Unset, str]):
        whole_hetjob (Union[Unset, str]):
        disable_whole_hetjob (Union[Unset, str]):
        disable_wait_for_result (Union[Unset, str]):
        usage_time_as_submit_time (Union[Unset, str]):
        show_batch_script (Union[Unset, str]):
        show_job_environment (Union[Unset, str]):
        format_ (Union[Unset, str]):
        groups (Union[Unset, str]):
        job_name (Union[Unset, str]):
        partition (Union[Unset, str]):
        qos (Union[Unset, str]):
        reason (Union[Unset, str]):
        reservation (Union[Unset, str]):
        reservation_id (Union[Unset, str]):
        state (Union[Unset, str]):
        step (Union[Unset, str]):
        end_time (Union[Unset, str]):
        start_time (Union[Unset, str]):
        node (Union[Unset, str]):
        users (Union[Unset, str]):
        wckey (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        V0043OpenapiSlurmdbdJobsResp
    """

    return (
        await asyncio_detailed(
            client=client,
            account=account,
            association=association,
            cluster=cluster,
            constraints=constraints,
            scheduler_unset=scheduler_unset,
            scheduled_on_submit=scheduled_on_submit,
            scheduled_by_main=scheduled_by_main,
            scheduled_by_backfill=scheduled_by_backfill,
            job_started=job_started,
            exit_code=exit_code,
            show_duplicates=show_duplicates,
            skip_steps=skip_steps,
            disable_truncate_usage_time=disable_truncate_usage_time,
            whole_hetjob=whole_hetjob,
            disable_whole_hetjob=disable_whole_hetjob,
            disable_wait_for_result=disable_wait_for_result,
            usage_time_as_submit_time=usage_time_as_submit_time,
            show_batch_script=show_batch_script,
            show_job_environment=show_job_environment,
            format_=format_,
            groups=groups,
            job_name=job_name,
            partition=partition,
            qos=qos,
            reason=reason,
            reservation=reservation,
            reservation_id=reservation_id,
            state=state,
            step=step,
            end_time=end_time,
            start_time=start_time,
            node=node,
            users=users,
            wckey=wckey,
        )
    ).parsed
