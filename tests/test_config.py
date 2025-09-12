from pathlib import Path

from pydantic import ValidationError
import pytest

from polar_flow.server.config import Config


def test_config_defaults_when_missing(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")  # empty file -> defaults and env fallback
    cfg = Config.load(cfg_path)
    assert cfg.server.secret_key  # defaulted
    assert cfg.server.database_url.startswith("sqlite:///")
    assert cfg.server.redis_url.startswith("redis://")
    assert cfg.server.scheduler_poll_interval > 0
    assert cfg.defaults.user_priority == 100


def test_config_invalid_interval_raises(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("""
[server]
secret_key="x"
database_url="sqlite:///x.db"
redis_url="redis://localhost:6379/0"
scheduler_poll_interval=0
""")
    with pytest.raises(ValidationError, match="greater than 0"):
        Config.load(cfg_path)
