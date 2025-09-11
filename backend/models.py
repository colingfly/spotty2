# backend/models.py
from __future__ import annotations
from sqlalchemy import Column, Integer, String, Text, Float, BigInteger, DateTime, UniqueConstraint, Index
from sqlalchemy.sql import func
from db import Base


class SpotifyAccount(Base):
    __tablename__ = "spotify_accounts"

    id = Column(Integer, primary_key=True)
    spotify_user_id = Column(String(128), unique=True, index=True, nullable=False)
    app_user_id = Column(String(128), nullable=True)

    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(BigInteger, nullable=False)

    scope = Column(Text, nullable=True)
    token_type = Column(String(32), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# --- New: cached user taste vector (audio features) + genre set ---
class UserTaste(Base):
    __tablename__ = "user_taste"

    id = Column(Integer, primary_key=True)
    spotify_user_id = Column(String(128), unique=True, index=True, nullable=False)
    updated_at = Column(BigInteger, default=0)  # unix seconds

    # averaged audio features
    danceability = Column(Float)
    energy = Column(Float)
    valence = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    speechiness = Column(Float)
    tempo = Column(Float)

    # CSV of lowercased genres
    genres = Column(Text, default="")


# --- New: simple scan log (optional analytics) ---
class PosterScan(Base):
    __tablename__ = "poster_scan"

    id = Column(Integer, primary_key=True)
    spotify_user_id = Column(String(128), index=True, nullable=False)
    scan_ts = Column(BigInteger, index=True)             # unix seconds
    artists_csv = Column(Text, default="")               # CSV of resolved names


# --- New: co-occurrence graph learned from posters ---
class ArtistEdge(Base):
    """
    Undirected co-occurrence between two resolved artist names.
    Store lowercased, sorted names (a <= b) for uniqueness.
    """
    __tablename__ = "artist_edge"

    id = Column(Integer, primary_key=True)
    a = Column(String(256), index=True, nullable=False)
    b = Column(String(256), index=True, nullable=False)
    weight = Column(Float, default=0.0)       # accumulate per poster scan
    last_seen = Column(BigInteger, default=0)

    __table_args__ = (
        UniqueConstraint("a", "b", name="uq_artist_edge_ab"),
        Index("ix_artist_edge_a_b", "a", "b"),
    )
