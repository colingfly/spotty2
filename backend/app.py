import os
import time
import base64
from urllib.parse import urlencode

import requests
from flask import Flask, request, redirect
from flask_cors import CORS
from dotenv import load_dotenv

# --- DB wiring ---
from db import SessionLocal
from models import SpotifyAccount

import io
from PIL import Image
import pytesseract
import cv2
import numpy as np
from rapidfuzz import fuzz

# ==================== Config ====================
load_dotenv()  # reads backend/.env

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8888"))

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
SCOPE = "user-top-read"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== Helpers ====================
# IMPORTANT: point pytesseract to Windows install
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_text_from_image(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10
    )
    pil = Image.fromarray(th)
    text = pytesseract.image_to_string(pil, lang="eng")
    return text


def tokenize_candidate_lines(text: str) -> list[str]:
    raw = [ln.strip() for ln in text.splitlines()]
    bad = {"doors", "live", "stage", "tickets", "am", "pm", "show", "venue"}
    keep = []
    for ln in raw:
        if not ln or len(ln) < 2:
            continue
        if any(ch.isdigit() for ch in ln) and len(ln) <= 4:
            continue
        clean = "".join(
            ch for ch in ln if ch.isalnum() or ch.isspace() or ch in "&-.'"
        )
        words = clean.split()
        if len(words) > 7:
            continue
        keep.append(clean.strip())
    seen, out = set(), []
    for k in keep:
        low = k.lower()
        if low not in seen:
            seen.add(low)
            out.append(k)
    return out


def spotify_search_artist(name: str, bearer: str) -> dict | None:
    r = requests.get(
        f"{API_BASE}/search",
        headers={"Authorization": f"Bearer {bearer}"},
        params={"q": name, "type": "artist", "limit": 3},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    items = (r.json().get("artists") or {}).get("items") or []
    return items[0] if items else None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def get_user_top_genres(acct: SpotifyAccount) -> set[str]:
    r = requests.get(
        f"{API_BASE}/me/top/artists",
        headers={"Authorization": f"Bearer {acct.access_token}"},
        params={"time_range": "long_term", "limit": 50},
        timeout=15,
    )
    if r.status_code != 200:
        return set()
    artists = r.json().get("items", [])
    genres: set[str] = set()
    for a in artists:
        for g in a.get("genres", []):
            genres.add(g.lower())
    return genres


def _basic_auth_header():
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _refresh_if_needed(acct: SpotifyAccount, db) -> SpotifyAccount:
    if time.time() < (acct.expires_at - 30):
        return acct
    if not acct.refresh_token:
        raise RuntimeError("Missing refresh token; user must re-authenticate.")

    r = requests.post(
        TOKEN_URL,
        headers=_basic_auth_header(),
        data={
            "grant_type": "refresh_token",
            "refresh_token": acct.refresh_token,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Refresh failed: {r.status_code} {r.text}")

    payload = r.json()
    acct.access_token = payload["access_token"]
    acct.expires_at = int(time.time()) + int(payload.get("expires_in", 3600))
    if "refresh_token" in payload:
        acct.refresh_token = payload["refresh_token"]
    acct.scope = payload.get("scope", acct.scope)
    acct.token_type = payload.get("token_type", acct.token_type)

    db.add(acct)
    db.commit()
    db.refresh(acct)
    return acct


# ==================== API: Scan Poster ====================
@app.post("/api/scan")
def api_scan():
    if "file" not in request.files:
        return {"error": "missing_file"}, 400
    sid = request.form.get("spotify_user_id")
    if not sid:
        return {"error": "missing_param", "message": "spotify_user_id required"}, 400

    db = next(_get_db())
    acct = db.query(SpotifyAccount).filter_by(spotify_user_id=sid).one_or_none()
    if not acct:
        return {"error": "not_linked"}, 404
    try:
        acct = _refresh_if_needed(acct, db)
    except Exception as e:
        return {"error": "refresh_failed", "detail": str(e)}, 401

    f = request.files["file"]
    file_bytes = f.read()

    text = ocr_text_from_image(file_bytes)
    candidates = tokenize_candidate_lines(text)
    user_genres = get_user_top_genres(acct)

    results = []
    for cand in candidates:
        found = spotify_search_artist(cand, acct.access_token)
        if not found:
            continue
        found_name = found["name"]
        name_score = fuzz.token_set_ratio(cand.lower(), found_name.lower())
        artist_genres = set(g.lower() for g in found.get("genres", []))
        genre_score = jaccard(artist_genres, user_genres)
        total = 0.6 * (name_score / 100.0) + 0.4 * genre_score

        results.append(
            {
                "candidate": cand,
                "resolved_name": found_name,
                "spotify_artist_id": found["id"],
                "image": (found.get("images") or [{}])[0].get("url"),
                "external_url": (found.get("external_urls") or {}).get("spotify"),
                "genres": sorted(artist_genres)[:4],
                "popularity": found.get("popularity"),
                "scores": {
                    "name": round(name_score, 1),
                    "genre": round(genre_score, 3),
                    "total": round(total, 3),
                },
            }
        )

    results.sort(key=lambda x: x["scores"]["total"], reverse=True)
    pruned = [r for r in results if r["scores"]["total"] >= 0.35][:20]

    return {
        "count": len(pruned),
        "items": pruned,
        "debug": {"candidates": candidates[:30]},
    }


# ==================== Basic Routes ====================
@app.get("/")
def root():
    return {"ok": True, "service": "spotty-backend"}


@app.get("/api/health/db")
def health_db():
    try:
        db = next(_get_db())
        db.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ==================== OAuth ====================
@app.get("/login")
def login():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "show_dialog": "false",
    }
    return redirect(f"{AUTH_URL}?{urlencode(params)}")


@app.get("/callback")
def callback():
    error = request.args.get("error")
    if error:
        return {"error": error}, 400
    code = request.args.get("code")
    if not code:
        return {"error": "missing_code"}, 400

    r = requests.post(
        TOKEN_URL,
        headers=_basic_auth_header(),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=15,
    )
    if r.status_code != 200:
        return {"error": "token_exchange_failed", "body": r.text}, r.status_code
    tokens = r.json()

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    expires_at = int(time.time()) + int(tokens.get("expires_in", 3600))

    me = requests.get(
        f"{API_BASE}/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if me.status_code != 200:
        return {"error": "me_failed", "body": me.text}, me.status_code
    user = me.json()
    spotify_user_id = user["id"]

    db = next(_get_db())
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

    return redirect(f"{FRONTEND_URL}/#sid={spotify_user_id}")


# ==================== API: Top Artists ====================
@app.get("/api/me/top-artists")
def api_top_artists():
    sid = request.args.get("spotify_user_id")
    if not sid:
        return {"error": "missing_param", "message": "spotify_user_id is required"}, 400

    time_range = request.args.get("time_range", "medium_term")
    try:
        limit = min(max(int(request.args.get("limit", 20)), 1), 50)
    except Exception:
        limit = 20

    db = next(_get_db())
    acct = db.query(SpotifyAccount).filter_by(spotify_user_id=sid).one_or_none()
    if not acct:
        return {"error": "not_linked", "message": "spotify_user_id not found in DB"}, 404

    try:
        acct = _refresh_if_needed(acct, db)
    except Exception as e:
        return {"error": "refresh_failed", "detail": str(e)}, 401

    r = requests.get(
        f"{API_BASE}/me/top/artists",
        headers={"Authorization": f"Bearer {acct.access_token}"},
        params={"time_range": time_range, "limit": limit},
        timeout=20,
    )

    if r.status_code == 401:
        return {"error": "unauthorized", "message": "Access token invalid/expired"}, 401
    if r.status_code == 403:
        return {"error": "forbidden", "message": "Missing scope user-top-read"}, 403
    if not r.ok:
        return {
            "error": "spotify_error",
            "status_code": r.status_code,
            "body": r.text,
        }, r.status_code

    items = (r.json() or {}).get("items", [])
    artists = [
        {
            "id": a.get("id"),
            "name": a.get("name"),
            "genres": a.get("genres", []),
            "popularity": a.get("popularity"),
            "image": (a.get("images") or [{}])[0].get("url"),
            "external_url": a.get("external_urls", {}).get("spotify"),
        }
        for a in items
    ]

    return {"items": artists, "count": len(artists), "limit": limit, "time_range": time_range}


# ==================== Run (dev) ====================
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True)
