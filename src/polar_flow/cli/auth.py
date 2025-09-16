from __future__ import annotations

import base64
import json
import time
from typing import cast

import httpx
import typer
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from polar_flow.cli.printers import print_debug, print_kv

from .config import AppConfig, save_token

app = typer.Typer(help="Authorization")


def _fetch_server_pubkey(client: httpx.Client, base_url: str) -> bytes:
    """从 ep /pubkey 获取公钥"""
    r = client.get(f"{base_url}/pubkey", timeout=10.0)
    r.raise_for_status()
    return r.content  # bytes of PEM


def _encrypt_payload(pub_pem: bytes, payload: dict) -> str:
    """利用 RSA-OAEP 和所给的公钥加密 payload"""
    pub = cast("rsa.RSAPublicKey", serialization.load_pem_public_key(pub_pem))
    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ciphertext = pub.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(ciphertext).decode("ascii")


@app.command("login")
def login(
    ctx: typer.Context,
    username: str | None = typer.Option(..., "--username", "-u", prompt=True, help="用户名"),
    password: str | None = typer.Option(
        ..., "--password", "-p", prompt=True, help="密码", hide_input=True,
    ),
) -> None:
    """登录并获取身份认证"""
    cfg: AppConfig = ctx.obj["cfg"]
    debug: bool = ctx.obj["debug"]
    base_url = f"http://{cfg.pam_server.host}:{cfg.pam_server.port}"

    print_debug(f"username: {username}, password: {password}", debug=debug)

    with httpx.Client() as client:
        # 1) 获取服务器公钥（PEM）
        pub_pem = _fetch_server_pubkey(client, base_url)

        # 2) 组装明文并加密
        payload = {"username": username, "password": password, "ts": int(time.time())}
        ciphertext_b64 = _encrypt_payload(pub_pem, payload)

        # 3) 发送密文获取 JWT
        resp = client.post(
            f"{base_url}/auth/token",
            json={"ciphertext_b64": ciphertext_b64},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

    token = data["access_token"]
    expires = int(data.get("expires_in", 3600))
    save_token(token, expires)
    print_kv(
        "成功认证",
        {
            "token": token,
            "expires": f"{expires} s",
        },
        cfg.logging.dict_style,
    )
