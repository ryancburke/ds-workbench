import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    databricks_host: str = field(default_factory=lambda: os.getenv("DATABRICKS_HOST", ""))
    databricks_token: str = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN", ""))
    databricks_http_path: str = field(default_factory=lambda: os.getenv("DATABRICKS_HTTP_PATH", ""))
    local_dev_mode: bool = field(
        default_factory=lambda: os.getenv("LOCAL_DEV_MODE", "true").lower() == "true"
    )


config = Config()
