import tomllib

from pydantic import BaseModel, Field


class FlaskConfig(BaseModel):
    debug: bool = Field(default=False)

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")

class PamServerConfig(BaseModel):
    host: str
    port: int

class SlurmServerConfig(BaseModel):
    host: str
    port: int

class AppConfig(BaseModel):
    flask: FlaskConfig
    logging: LoggingConfig
    pam_server: PamServerConfig = Field(alias="pam-server")
    slurm_server: SlurmServerConfig = Field(alias="slurm-server")

def load_config(path: str) -> AppConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return AppConfig(**data)


if __name__ == "__main__":
    cfg = load_config("data/prod.toml")
    print(cfg.model_dump())
    print(cfg.pam_server.host)
