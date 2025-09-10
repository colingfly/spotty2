# backend/models.py
from sqlalchemy import Column, Integer, String, BigInteger, Index
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime
from db import Base

class SpotifyAccount(Base):
    __tablename__ = "spotify_accounts"

    id = Column(Integer, primary_key=True)
    spotify_user_id = Column(String(128), unique=True, index=True, nullable=False)
    app_user_id = Column(String(128), nullable=True)

    access_token = Column(String(2048), nullable=False)
    refresh_token = Column(String(2048), nullable=True)
    expires_at = Column(BigInteger, nullable=False)
    scope = Column(String(512), nullable=True)
    token_type = Column(String(64), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

Index("ix_spotify_accounts_user", SpotifyAccount.spotify_user_id)
