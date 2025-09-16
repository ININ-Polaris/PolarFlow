from __future__ import annotations

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
            r.raise_for_status()
        except httpx.HTTPStatusError:
            if code == 401:
                print_error("未登录，请重新登录或检查网络连接", "Unauthorized")
            elif code == 511:
                print_error(
                    "认证失败或权限不足，请重新登录并确认权限正确",
                    "Network Authentication Required",
                )
            elif not self._debug:
                print_error("请联系管理员，或使用 --debug", "未知错误")

            if self._debug:
                raise
            raise typer.Exit(1) from None

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.get(url, headers=self._headers(), params=params)
        print_debug(
            f"url: {url}\ncmd: curl -X GET {url} -H 'X-SLURM-USER-TOKEN: {self._token}'",
            "GET",
            debug=self._debug,
        )
        self._error_handler(r)
        return r.json()

    def delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.delete(url, headers=self._headers(), params=params)
        self._error_handler(r)
        print_debug(f"url: {url}", "DELETE", debug=self._debug)
        return r.json()

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.post(
            url,
            headers=self._headers() | {"Content-Type": "application/json"},
            json=body,
        )
        print_debug(f"url: {url}", "POST", debug=self._debug)
        self._error_handler(r)
        return r.json()
