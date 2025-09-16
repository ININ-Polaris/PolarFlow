import json
import os
import tomllib
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
    print(data)
    return AppConfig(**data)


def save_token(token: str, expires_in: int) -> None:
    TOKEN_PATH.write_text(json.dumps({"token": token, "expires_in": expires_in}, indent=2))


def load_token() -> str | None:
    if TOKEN_PATH.exists():
        try:
            return str(json.loads(TOKEN_PATH.read_text()).get("token"))
        except Exception:  # noqa: BLE001
            return None
    return None


if __name__ == "__main__":
    cfg = load_config(Path("data/prod.toml"))
    print(cfg.model_dump())
    print(cfg.pam_server.host)
