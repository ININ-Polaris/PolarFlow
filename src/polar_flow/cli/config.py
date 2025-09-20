import base64
import json
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STATE_DIR = Path(os.environ.get("POLAR_CONFIG_PATH", "~/.config/polarflow")).expanduser()
STATE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_PATH = STATE_DIR / "token.json"


class PamServerConfig(BaseModel):
    host: str
    port: int


class SlurmServerConfig(BaseModel):
    host: str
    port: int


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    dict_style: Literal["table", "dict"] = Field(alias="dict-style", default="table")


class AppConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    pam_server: PamServerConfig = Field(alias="pam-server")
    slurm_server: SlurmServerConfig = Field(alias="slurm-server")
    logging: LoggingConfig = LoggingConfig()


def load_config(path: Path) -> AppConfig:
    p = path.expanduser()
    if not p.exists():
        raise FileNotFoundError

    with open(p, "rb") as f:
        data = tomllib.load(f)
    return AppConfig(**data)


@dataclass
class Token:
    jwt: str
    expires_in: int

    def to_dict(self) -> dict[str, str | int]:
        return {"token": self.jwt, "expires_in": self.expires_in}

    @staticmethod
    def from_dict(text: str) -> "Token":
        data = json.loads(text)
        return Token(jwt=data.get("token", ""), expires_in=int(data.get("expires_in", 0)))


def save_token(token: Token) -> None:
    TOKEN_PATH.write_text(json.dumps(token.to_dict(), indent=2))


def load_token() -> Token | None:
    if TOKEN_PATH.exists():
        try:
            return Token.from_dict(TOKEN_PATH.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def is_jwt_expired(token: Token) -> bool:
    jwt = token.jwt
    try:
        payload_b64 = jwt.split(".")[1]
        payload = json.loads(b64url_decode(payload_b64))
        exp = payload.get("exp")
        return exp is not None and time.time() >= int(exp)
    except Exception:  # noqa: BLE001
        return True


if __name__ == "__main__":
    cfg = load_config(Path("data/prod.toml"))
    print(cfg.model_dump())
    print(cfg.pam_server.host)
