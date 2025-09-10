# main.py
import os, time, base64, secrets
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.db import SessionLocal, engine
from backend.models import Base, SpotifyAccount

load_dotenv()
Base.metadata.create_all(bind=engine)

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
SCOPE = "user-top-read"

app = FastAPI(title="Spotty Backend")

# CORS for local mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simple state store (swap to Redis if you want)
STATE_STORE = {}

def _basic_auth_header():
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}

async def _refresh_if_needed(acct: SpotifyAccount, db: Session):
    if not acct.is_expired():
        return acct
    if not acct.refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token; re-login required")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            TOKEN_URL,
            headers=_basic_auth_header(),
            data={
                "grant_type": "refresh_token",
                "refresh_token": acct.refresh_token,
                "redirect_uri": REDIRECT_URI,
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    payload = resp.json()
    acct.access_token = payload["access_token"]
    acct.expires_at = int(time.time()) + int(payload.get("expires_in", 3600))
    if "refresh_token" in payload:
        acct.refresh_token = payload["refresh_token"]
    acct.token_type = payload.get("token_type", acct.token_type)
    acct.scope = payload.get("scope", acct.scope)
    db.add(acct)
    db.commit()
    db.refresh(acct)
    return acct

@app.get("/")
def root():
    return {"ok": True, "message": "Spotty backend running"}

@app.get("/login")
def login():
    state = secrets.token_urlsafe(24)
    STATE_STORE[state] = int(time.time())

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "state": state,
        # For mobile PKCE in future: add code_challenge & code_challenge_method=S256
    }
    url = AUTH_URL + "?" + "&".join(f"{k}={httpx.QueryParams({k:v})[k]}" for k, v in params.items())
    return RedirectResponse(url)

@app.get("/callback")
async def callback(code: Optional[str] = None, state: Optional[str] = None, db: Session = Depends(get_db)):
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")
    if not state or state not in STATE_STORE:
        raise HTTPException(status_code=400, detail="Invalid state")

    # Exchange code -> tokens
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            TOKEN_URL,
            headers=_basic_auth_header(),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    tokens = resp.json()

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    expires_at = int(time.time()) + int(tokens.get("expires_in", 3600))

    # Get the Spotify user id
    async with httpx.AsyncClient(timeout=15) as client:
        me = await client.get(f"{API_BASE}/me", headers={"Authorization": f"Bearer {access_token}"})
    if me.status_code != 200:
        raise HTTPException(status_code=me.status_code, detail=me.text)
    user = me.json()
    spotify_user_id = user["id"]

    # Upsert account
    acct = db.query(SpotifyAccount).filter_by(spotify_user_id=spotify_user_id).one_or_none()
    if not acct:
        acct = SpotifyAccount(
            spotify_user_id=spotify_user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scope=tokens.get("scope"),
            token_type=tokens.get("token_type"),
        )
    else:
        acct.access_token = access_token
        acct.refresh_token = refresh_token or acct.refresh_token
        acct.expires_at = expires_at
        acct.scope = tokens.get("scope", acct.scope)
        acct.token_type = tokens.get("token_type", acct.token_type)

    db.add(acct)
    db.commit()
    db.refresh(acct)

    # For dev, just show a success JSON. In production, redirect back to your app.
    return JSONResponse({"ok": True, "spotify_user_id": spotify_user_id})

@app.get("/me/top-artists")
async def top_artists(spotify_user_id: str, time_range: str = "medium_term", limit: int = 20, db: Session = Depends(get_db)):
    acct = db.query(SpotifyAccount).filter_by(spotify_user_id=spotify_user_id).one_or_none()
    if not acct:
        raise HTTPException(status_code=404, detail="Spotify account not linked")

    acct = await _refresh_if_needed(acct, db)

    params = {"time_range": time_range, "limit": min(max(limit, 1), 50)}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{API_BASE}/me/top/artists",
            headers={"Authorization": f"Bearer {acct.access_token}"},
            params=params,
        )
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    # return a compact shape; switch to full payload if you want
    return {
        "count": len(data.get("items", [])),
        "items": [
            {
                "name": a["name"],
                "id": a["id"],
                "genres": a.get("genres", [])[:3],
                "url": a["external_urls"]["spotify"],
                "image": (a["images"][0]["url"] if a.get("images") else None),
            }
            for a in data.get("items", [])
        ],
    }
