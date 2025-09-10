# backend/config.py
from __future__ import annotations
import os, secrets
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def _safe_int(val: Optional[str], default: int) -> int:
    try:
        return int(str(val))
    except (TypeError, ValueError):
        return default

def _csv(s: Optional[str], default: List[str]) -> List[str]:
    if not s:
        return default
    return [part.strip() for part in s.split(",") if part.strip()]

@dataclass(frozen=True)
class Settings:
    # App
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = _safe_int(os.getenv("APP_PORT", "8888"), 8888)
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "") or secrets.token_hex(32)

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Spotify OAuth
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    spotify_redirect_uri: str = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    spotify_scope: str = os.getenv("SPOTIFY_SCOPE", "user-top-read")

    # Frontend + CORS
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    cors_origins: List[str] = None  # set in __post_init__

    # OCR
    tesseract_cmd: Optional[str] = os.getenv("TESSERACT_CMD")

    # Misc
    max_upload_mb: int = _safe_int(os.getenv("MAX_UPLOAD_MB", "8"), 8)

    def __post_init__(self):
        # dataclass(frozen=True) workaround to set computed field
        object.__setattr__(self, "cors_origins",
            _csv(os.getenv("CORS_ALLOW_ORIGINS"), [self.frontend_url])
        )

settings = Settings()
