from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

import httpx
import typer

from polar_flow._vendor.slurm_client import Client as SlurmGenClient
from polar_flow._vendor.slurm_client.api.slurm import (
    slurm_v0043_delete_job,
    slurm_v0043_get_diag,
    slurm_v0043_get_job,
    slurm_v0043_get_jobs,
    slurm_v0043_get_ping,
    slurm_v0043_post_job_submit,
)
from polar_flow._vendor.slurm_client.models.slurm_v0043_get_jobs_flags import SlurmV0043GetJobsFlags
from polar_flow._vendor.slurm_client.types import UNSET
from polar_flow.cli.printers import print_debug, print_error

if TYPE_CHECKING:
    from polar_flow._vendor.slurm_client.models.slurm_v0043_delete_job_flags import (
        SlurmV0043DeleteJobFlags,
    )
    from polar_flow._vendor.slurm_client.models.slurm_v0043_get_job_flags import (
        SlurmV0043GetJobFlags,
    )
    from polar_flow._vendor.slurm_client.models.v0043_job_submit_req import V0043JobSubmitReq
    from polar_flow._vendor.slurm_client.models.v0043_openapi_diag_resp import V0043OpenapiDiagResp
    from polar_flow._vendor.slurm_client.models.v0043_openapi_job_info_resp import (
        V0043OpenapiJobInfoResp,
    )
    from polar_flow._vendor.slurm_client.models.v0043_openapi_job_submit_response import (
        V0043OpenapiJobSubmitResponse,
    )
    from polar_flow._vendor.slurm_client.models.v0043_openapi_kill_job_resp import (
        V0043OpenapiKillJobResp,
    )
    from polar_flow._vendor.slurm_client.models.v0043_openapi_ping_array_resp import (
        V0043OpenapiPingArrayResp,
    )
    from polar_flow._vendor.slurm_client.types import Response

    from .config import AppConfig

DEFAULT_TIMEOUT = 10.0

ERROR_MESSAGES = {
    400: ("请求错误，请联系管理员、检查配置或网络连接", None),
    401: ("未登录，请重新登录或检查网络连接", None),
    404: ("请检查 ID NAME 等是否正确", None),
    511: ("认证失败或权限不足，请重新登录并确认权限正确", "Network Authentication Required"),
}


class SlurmClient:
    def __init__(self, cfg: AppConfig, token: str, debug: bool = False):
        self.base_url = f"http://{cfg.slurm_server.host}:{cfg.slurm_server.port}"

        self._debug = debug

        self._client = SlurmGenClient(
            base_url=self.base_url,
            headers={"Accept": "application/json", "X-SLURM-USER-TOKEN": token}
            if token
            else {"Accept": "application/json"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT),
            verify_ssl=True,  # 视实际需要
        )

    @staticmethod
    def _build_curl(
        method: str,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> str:
        # 基础命令
        cmd = ["curl", "-X", method.upper(), shlex.quote(url)]

        # 添加 headers
        for k, v in headers.items():
            cmd += ["-H", shlex.quote(f"{k}: {v}")]

        # 添加 params（仅打印用，不影响请求）
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            cmd[2] = shlex.quote(f"{url}?{query}")

        # 添加 body
        if body:
            cmd += ["-d", shlex.quote(json.dumps(body, ensure_ascii=False))]

        return " ".join(cmd)

    def _error_handler[T](self, res: Response[T]) -> T:
        if res.parsed is None:
            print_error("请检查网络连接", "操作失败")
            raise typer.Exit(1) from None
        payload = res.parsed
        status_code = res.status_code
        raw_text = res.content.decode()
        try:
            if 200 <= status_code < 300:
                return payload
            if status_code == 500 and isinstance(payload, dict) and "errors" in payload:
                error = payload["errors"][0]
                print_error(
                    error["error"],
                    title=f"{error['description']} [{error['error_number']}]",
                )
            elif status_code in ERROR_MESSAGES:
                msg, title = ERROR_MESSAGES[status_code]
                print_error(msg, title or f"HTTP {status_code}")
            elif not self._debug:
                # print_error("请联系管理员，或使用 --debug", "未知错误")
                return payload
        finally:
            if raw_text:
                print_debug(raw_text, debug=self._debug, title="Raw Response")
        return payload

    def get_diag(self) -> V0043OpenapiDiagResp:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/diag"
            print_debug(
                self._build_curl("GET", url=url, headers=self._client._headers),
                debug=self._debug,
            )
        res = slurm_v0043_get_diag.sync_detailed(client=self._client)
        return self._error_handler(res)

    def get_ping(self) -> V0043OpenapiPingArrayResp:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/ping"
            print_debug(
                self._build_curl("GET", url=url, headers=self._client._headers),
                debug=self._debug,
            )
        res = slurm_v0043_get_ping.sync_detailed(client=self._client)
        return self._error_handler(res)

    def get_jobs(
        self,
        update_time: None | str,
        flags: None | SlurmV0043GetJobsFlags,
    ) -> V0043OpenapiJobInfoResp:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/jobs"
            print_debug(
                self._build_curl("GET", url=url, headers=self._client._headers),
                debug=self._debug,
            )
        res = slurm_v0043_get_jobs.sync_detailed(
            client=self._client,
            update_time=update_time if update_time else UNSET,
            flags=flags if flags else SlurmV0043GetJobsFlags.ALL,
        )
        return self._error_handler(res)

    def get_job(
        self,
        job_id: str,
        update_time: None | str,
        flags: None | SlurmV0043GetJobFlags,
    ) -> V0043OpenapiJobInfoResp:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/job/{job_id}"
            print_debug(
                self._build_curl("GET", url=url, headers=self._client._headers),
                debug=self._debug,
            )
        res = slurm_v0043_get_job.sync_detailed(
            job_id=job_id,
            client=self._client,
            update_time=update_time if update_time else UNSET,
            flags=flags if flags else UNSET,
        )
        return self._error_handler(res)

    def delete_job(
        self,
        job_id: str,
        signal: None | str,
        flags: None | SlurmV0043DeleteJobFlags,
    ) -> V0043OpenapiKillJobResp:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/job/{job_id}"
            print_debug(
                self._build_curl(
                    "DELETE",
                    url=url,
                    headers=self._client._headers,
                    params={
                        "signal": signal,
                        "flags": flags,
                    },
                ),
                debug=self._debug,
            )
        res = slurm_v0043_delete_job.sync_detailed(
            job_id=job_id,
            client=self._client,
            signal=signal if signal else UNSET,
            flags=flags if flags else UNSET,
        )
        return self._error_handler(res)

    def submit_job(
        self,
        body: V0043JobSubmitReq,
    ) -> V0043OpenapiJobSubmitResponse:
        if self._debug:
            url = f"{self.base_url}/slurm/v0.0.43/job/submit"
            print_debug(
                self._build_curl(
                    "POST",
                    url=url,
                    headers=self._client._headers,
                    body=body.to_dict(),
                ),
                debug=self._debug,
            )
        res = slurm_v0043_post_job_submit.sync_detailed(
            client=self._client,
            body=body,
        )
        return self._error_handler(res)
