from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

import httpx
import typer

from polar_flow.cli.printers import print_debug, print_error

if TYPE_CHECKING:
    from .config import AppConfig

DEFAULT_TIMEOUT = 10.0


class SlurmClient:
    def __init__(self, cfg: AppConfig, token: str, debug: bool = False, prefix: str = "slurm"):
        self.base_url = f"http://{cfg.slurm_server.host}:{cfg.slurm_server.port}/{prefix}/v0.0.43"
        self._token = token
        self._client = httpx.Client(timeout=DEFAULT_TIMEOUT)
        self._debug = debug

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["X-SLURM-USER-TOKEN"] = self._token
        return headers

    def _error_handler(self, r: httpx.Response) -> None:
        try:
            code = r.status_code
            content = json.loads(r.content.decode())
            r.raise_for_status()
        except httpx.HTTPStatusError:
            if code == 401:
                print_error("未登录，请重新登录或检查网络连接", "Unauthorized")
            elif code == 404:
                print_error("请检查 ID NAME 等是否正确", "资源不存在")
            elif code == 511:
                print_error(
                    "认证失败或权限不足，请重新登录并确认权限正确",
                    "Network Authentication Required",
                )
            elif code == 500:
                error = content["errors"][0]
                print_error(
                    error["error"],
                    title=f"{error['description']} [{error['error_number']}]",
                )
            elif not self._debug:
                print_error("请联系管理员，或使用 --debug", "未知错误")
            else:
                raise

            if self._debug:
                print_error(r.content.decode())

            raise typer.Exit(1) from None

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

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.get(url, headers=self._headers(), params=params)
        if self._debug:
            print_debug(
                self._build_curl("GET", url, self._headers(), params=params), debug=self._debug
            )
        self._error_handler(r)
        return r.json()

    def delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.delete(url, headers=self._headers(), params=params)
        if self._debug:
            print_debug(
                self._build_curl("DELETE", url, self._headers(), params=params), debug=self._debug
            )
        self._error_handler(r)
        return r.json()

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._headers() | {"Content-Type": "application/json"}
        r = self._client.post(url, headers=headers, json=body)
        if self._debug:
            print_debug(self._build_curl("POST", url, headers, body=body), debug=self._debug)
        self._error_handler(r)
        return r.json()
