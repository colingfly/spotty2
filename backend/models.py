# backend/models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    DateTime,
    func,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# -------------------------
# spotify_accounts
# -------------------------
class SpotifyAccount(Base):
    __tablename__ = "spotify_accounts"

    id = Column(Integer, primary_key=True, index=True)
    # Spotify user id (e.g. "colingfly")
    spotify_user_id = Column(String, index=True, nullable=False)

    # Optional application-scoped user id (if you ever map multiple auth providers)
    app_user_id = Column(String, nullable=True)

    # OAuth tokens
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    # unix epoch seconds for expiry (Postgres int8)
    expires_at = Column(BigInteger, nullable=False)

    scope = Column(Text, nullable=True)
    token_type = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_spotify_accounts_spotify_user_id", "spotify_user_id"),
    )

    def __repr__(self) -> str:
        return f"<SpotifyAccount user={self.spotify_user_id} expires_at={self.expires_at}>"


# -------------------------
# user_taste
# -------------------------
class UserTaste(Base):
    __tablename__ = "user_taste"

    id = Column(Integer, primary_key=True, index=True)
    spotify_user_id = Column(String, index=True, nullable=False)

    # timestamptz in DB
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # aggregated audio features
    danceability      = Column(Float, nullable=False, default=0.0)
    energy            = Column(Float, nullable=False, default=0.0)
    valence           = Column(Float, nullable=False, default=0.0)
    acousticness      = Column(Float, nullable=False, default=0.0)
    instrumentalness  = Column(Float, nullable=False, default=0.0)
    liveness          = Column(Float, nullable=False, default=0.0)
    speechiness       = Column(Float, nullable=False, default=0.0)
    tempo             = Column(Float, nullable=False, default=0.0)

    # jsonb array of strings (e.g., ["indie rock","alt pop",...])
    genres = Column(JSONB, nullable=False, default=list)

    __table_args__ = (
        Index("ix_user_taste_spotify_user_id", "spotify_user_id"),
    )

    def __repr__(self) -> str:
        return f"<UserTaste user={self.spotify_user_id} updated_at={self.updated_at}>"


# -------------------------
# poster_scan (optional log)
# -------------------------
class PosterScan(Base):
    __tablename__ = "poster_scan"

    id = Column(Integer, primary_key=True, index=True)
    spotify_user_id = Column(String, index=True, nullable=False)

    # unix epoch seconds (int8) when scan happened
    scan_ts = Column(BigInteger, nullable=False)

    # CSV of artist names detected (if you log it)
    artists_csv = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_poster_scan_spotify_user_id", "spotify_user_id"),
    )

    def __repr__(self) -> str:
        return f"<PosterScan user={self.spotify_user_id} ts={self.scan_ts}>"


# -------------------------
# artist_edge (co-occurrence graph)
# -------------------------
class ArtistEdge(Base):
    __tablename__ = "artist_edge"

    id = Column(Integer, primary_key=True, index=True)
    # Undirected pair (store lexicographically a <= b in code)
    a = Column(String, index=True, nullable=False)
    b = Column(String, index=True, nullable=False)

    # strength of association
    weight = Column(Float, nullable=False, default=0.0)

    # last time we observed this co-occurrence (epoch seconds)
    last_seen = Column(BigInteger, nullable=False, default=0)

    __table_args__ = (
        Index("ix_artist_edge_pair", "a", "b", unique=True),
    )

    def __repr__(self) -> str:
        return f"<ArtistEdge {self.a}—{self.b} w={self.weight}>"
