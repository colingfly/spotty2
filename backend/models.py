# backend/models.py

from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime,
    func, UniqueConstraint, Index
)
from sqlalchemy.orm import Mapped, mapped_column
from db import Base


class SpotifyAccount(Base):
    __tablename__ = "spotify_accounts"

    # Surrogate PK
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Natural key from Spotify
    spotify_user_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)

    # (Optional) If you later link to your own users table
    app_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # OAuth tokens
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Epoch seconds (when access_token expires)
    expires_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Bookkeeping from Spotify
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Timestamps (server-side)
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("spotify_user_id", name="uq_spotify_accounts_user"),
        Index("ix_spotify_accounts_app_user_id", "app_user_id"),
    )

    def __repr__(self) -> str:
        return f"<SpotifyAccount spotify_user_id={self.spotify_user_id!r}>"
