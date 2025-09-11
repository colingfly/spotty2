# backend/app.py
from __future__ import annotations

import base64
import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Generator, Iterable, List, Dict, Any

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image, UnidentifiedImageError
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode
from urllib3.util.retry import Retry

from flask import Flask, jsonify, redirect, request
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from db import SessionLocal
from models import SpotifyAccount, UserTaste, PosterScan, ArtistEdge

# ---------------------------
# Logging
# ---------------------------
log = logging.getLogger("spotty")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Constants
# ---------------------------
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

# ---------------------------
# HTTP client with retries
# ---------------------------
def _spotify_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.timeout = 15
    return s


# ---------------------------
# Flask factory
# ---------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)

    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024

    CORS(
        app,
        resources={r"/api/*": {"origins": settings.cors_origins}},
        supports_credentials=False,
    )

    # Optional Tesseract override (Windows dev boxes)
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_db() -> Generator[Session, None, None]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _basic_auth_header() -> Dict[str, str]:
        raw = f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}

    def _refresh_if_needed(acct: SpotifyAccount, db: Session) -> SpotifyAccount:
        # Refresh 30s before expiry
        if time.time() < (acct.expires_at - 30):
            return acct
        if not acct.refresh_token:
            raise RuntimeError("Missing refresh token; user must re-auth.")

        sess = _spotify_session()
        r = sess.post(
            TOKEN_URL,
            headers=_basic_auth_header(),
            data={
                "grant_type": "refresh_token",
                "refresh_token": acct.refresh_token,
                "redirect_uri": settings.spotify_redirect_uri,
            },
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

    # ------------ OCR ------------
    def _ocr_text_from_image(file_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("Not a valid image file.")

        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10
        )
        pil = Image.fromarray(th)
        try:
            return pytesseract.image_to_string(pil, lang="eng")
        except (OSError, pytesseract.TesseractNotFoundError) as e:
            raise RuntimeError("OCR unavailable on this server") from e

    def _tokenize_candidate_lines(text_blob: str) -> List[str]:
        raw = [ln.strip() for ln in text_blob.splitlines()]
        keep: List[str] = []
        bad = {"doors", "live", "stage", "tickets", "am", "pm", "show", "venue"}

        for ln in raw:
            if not ln or len(ln) < 2:
                continue
            # very short numeric lines or times
            if any(ch.isdigit() for ch in ln) and len(ln) <= 4:
                continue
            # clean punctuation (keep pragmatic separators)
            clean = "".join(
                ch for ch in ln if ch.isalnum() or ch.isspace() or ch in "&-.'"
            )
            if not clean:
                continue
            words = clean.split()
            if len(words) > 7:
                continue
            if clean.lower() in bad:
                continue
            keep.append(clean.strip())

        # de-dupe (case-insensitive)
        seen = set()
        out: List[str] = []
        for k in keep:
            low = k.lower()
            if low not in seen:
                seen.add(low)
                out.append(k)
        return out

    def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    # ------------ Spotify helpers ------------
    def _spotify_search_artist(name: str, bearer: str) -> Dict[str, Any] | None:
        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/search",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"q": name, "type": "artist", "limit": 3},
        )
        if r.status_code != 200:
            return None
        items = (r.json().get("artists") or {}).get("items") or []
        return items[0] if items else None

    def _user_top_artists(acct: SpotifyAccount, time_range="long_term", limit=50) -> List[Dict[str, Any]]:
        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/me/top/artists",
            headers={"Authorization": f"Bearer {acct.access_token}"},
            params={"time_range": time_range, "limit": limit},
        )
        if r.status_code != 200:
            return []
        return r.json().get("items", [])

    def _user_top_genres(acct: SpotifyAccount) -> List[str]:
        artists = _user_top_artists(acct, time_range="long_term", limit=50)
        genres: List[str] = []
        for a in artists:
            for g in a.get("genres", []):
                genres.append(g.lower())
        # keep unique but order stable
        seen = set()
        out: List[str] = []
        for g in genres:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    def _user_audio_features_profile(acct: SpotifyAccount) -> Dict[str, float]:
        """Average audio features across user's top tracks."""
        sess = _spotify_session()

        # Get top tracks (fallback to artists' top tracks if needed)
        r1 = sess.get(
            f"{API_BASE}/me/top/tracks",
            headers={"Authorization": f"Bearer {acct.access_token}"},
            params={"time_range": "medium_term", "limit": 50},
        )
        tracks = r1.json().get("items", []) if r1.status_code == 200 else []

        if not tracks:
            # fallback: pick user's top artists and fetch each artist's top tracks
            artists = _user_top_artists(acct, "medium_term", limit=10)
            for a in artists:
                rid = a.get("id")
                if not rid:
                    continue
                rr = sess.get(
                    f"{API_BASE}/artists/{rid}/top-tracks",
                    headers={"Authorization": f"Bearer {acct.access_token}"},
                    params={"market": "US"},
                )
                if rr.status_code == 200:
                    tracks.extend([{"id": t.get("id")} for t in rr.json().get("tracks", [])])

        track_ids = [t.get("id") for t in tracks if t.get("id")]
        if not track_ids:
            return {
                "danceability": 0.0, "energy": 0.0, "valence": 0.0,
                "acousticness": 0.0, "instrumentalness": 0.0,
                "liveness": 0.0, "speechiness": 0.0, "tempo": 0.0,
            }

        feats: List[Dict[str, Any]] = []
        for i in range(0, len(track_ids), 100):
            chunk = track_ids[i : i + 100]
            rr = sess.get(
                f"{API_BASE}/audio-features",
                headers={"Authorization": f"Bearer {acct.access_token}"},
                params={"ids": ",".join(chunk)},
            )
            if rr.status_code == 200:
                feats.extend(rr.json().get("audio_features", []) or [])

        keys = [
            "danceability", "energy", "valence",
            "acousticness", "instrumentalness", "liveness",
            "speechiness", "tempo",
        ]
        agg = {k: 0.0 for k in keys}
        n = 0
        for f in feats:
            if not f:
                continue
            n += 1
            for k in keys:
                v = f.get(k)
                if isinstance(v, (int, float)):
                    agg[k] += float(v)
        if n == 0:
            return {k: 0.0 for k in keys}
        return {k: round(agg[k] / n, 6) for k in keys}

    # ------------ Taste profile upsert ------------
    def _get_user_taste(acct: SpotifyAccount, db: Session) -> UserTaste:
        ut = (
            db.query(UserTaste)
            .filter_by(spotify_user_id=acct.spotify_user_id)
            .one_or_none()
        )
        if ut:
            return ut

        # Build fresh
        genres = _user_top_genres(acct)
        audio = _user_audio_features_profile(acct)
        ut = UserTaste(
            spotify_user_id=acct.spotify_user_id,
            updated_at=datetime.now(timezone.utc),  # timestamptz
            danceability=audio["danceability"],
            energy=audio["energy"],
            valence=audio["valence"],
            acousticness=audio["acousticness"],
            instrumentalness=audio["instrumentalness"],
            liveness=audio["liveness"],
            speechiness=audio["speechiness"],
            tempo=audio["tempo"],
            genres=genres,  # JSONB list
        )
        db.add(ut)
        db.commit()
        db.refresh(ut)
        return ut

    def _update_user_taste_if_stale(acct: SpotifyAccount, db: Session, max_age_hours=24) -> UserTaste:
        ut = (
            db.query(UserTaste)
            .filter_by(spotify_user_id=acct.spotify_user_id)
            .one_or_none()
        )
        now = datetime.now(timezone.utc)
        if ut and ut.updated_at and (now - ut.updated_at).total_seconds() < max_age_hours * 3600:
            return ut

        genres = _user_top_genres(acct)
        audio = _user_audio_features_profile(acct)
        if ut:
            ut.updated_at = now
            ut.genres = genres
            ut.danceability = audio["danceability"]
            ut.energy = audio["energy"]
            ut.valence = audio["valence"]
            ut.acousticness = audio["acousticness"]
            ut.instrumentalness = audio["instrumentalness"]
            ut.liveness = audio["liveness"]
            ut.speechiness = audio["speechiness"]
            ut.tempo = audio["tempo"]
            db.add(ut)
        else:
            ut = UserTaste(
                spotify_user_id=acct.spotify_user_id,
                updated_at=now,
                genres=genres,
                **audio,
            )
            db.add(ut)
        db.commit()
        db.refresh(ut)
        return ut

    # ------------ Artist co-occurrence ------------
    def _log_cooccurrence(names: List[str], db: Session) -> None:
        # Normalize and unique
        norm = [n.strip() for n in names if n and n.strip()]
        seen = set()
        uniq = []
        for n in norm:
            low = n.lower()
            if low not in seen:
                seen.add(low)
                uniq.append(n)

        # upsert undirected edges a<=b
        def keypair(a: str, b: str) -> tuple[str, str]:
            return (a, b) if a.lower() <= b.lower() else (b, a)

        epoch = int(time.time())

        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = keypair(uniq[i], uniq[j])
                edge = (
                    db.query(ArtistEdge)
                    .filter(ArtistEdge.a == a, ArtistEdge.b == b)
                    .one_or_none()
                )
                if not edge:
                    edge = ArtistEdge(a=a, b=b, weight=1.0, last_seen=epoch)
                else:
                    edge.weight = min(edge.weight + 1.0, 20.0)
                    edge.last_seen = epoch
                db.add(edge)
        db.commit()

    def _cooccur_score(cand_name: str, user_top_names: List[str], db: Session) -> float:
        """Simple sum of weights to user's top artist names."""
        if not user_top_names:
            return 0.0
        total = 0.0
        for t in user_top_names:
            a, b = (cand_name, t) if cand_name.lower() <= t.lower() else (t, cand_name)
            edge = (
                db.query(ArtistEdge)
                .filter(ArtistEdge.a == a, ArtistEdge.b == b)
                .one_or_none()
            )
            if edge:
                total += edge.weight
        # squash to 0..1 range (heuristic)
        return min(total / 20.0, 1.0)

    # ---------------------------
    # Routes
    # ---------------------------

    @app.get("/")
    def root():
        return {"ok": True, "service": "spotty-backend"}

    @app.get("/api/health/live")
    def live():
        return {"ok": True}

    @app.get("/api/health/ready")
    def ready():
        try:
            db = next(_get_db())
            db.execute(text("SELECT 1"))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    @app.get("/login")
    def login():
        params = {
            "client_id": settings.spotify_client_id,
            "response_type": "code",
            "redirect_uri": settings.spotify_redirect_uri,
            "scope": settings.spotify_scope,
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

        sess = _spotify_session()
        r = sess.post(
            TOKEN_URL,
            headers=_basic_auth_header(),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.spotify_redirect_uri,
            },
        )
        if r.status_code != 200:
            return {"error": "token_exchange_failed", "body": r.text}, r.status_code

        tokens = r.json()
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")
        expires_at = int(time.time()) + int(tokens.get("expires_in", 3600))

        me = sess.get(
            f"{API_BASE}/me", headers={"Authorization": f"Bearer {access_token}"}
        )
        if me.status_code != 200:
            return {"error": "me_failed", "body": me.text}, me.status_code
        user = me.json()
        spotify_user_id = user["id"]

        db = next(_get_db())
        acct = (
            db.query(SpotifyAccount)
            .filter_by(spotify_user_id=spotify_user_id)
            .one_or_none()
        )
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

        return redirect(f"{settings.frontend_url}/#sid={spotify_user_id}")

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
            return {"error": "not_linked"}, 404
        try:
            acct = _refresh_if_needed(acct, db)
        except Exception as e:
            return {"error": "refresh_failed", "detail": str(e)}, 401

        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/me/top/artists",
            headers={"Authorization": f"Bearer {acct.access_token}"},
            params={"time_range": time_range, "limit": limit},
        )
        if r.status_code == 401:
            return {"error": "unauthorized"}, 401
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
        return {
            "items": artists,
            "count": len(artists),
            "limit": limit,
            "time_range": time_range,
        }

    @app.post("/api/scan")
    def api_scan():
        f = request.files.get("file")
        sid = request.form.get("spotify_user_id")
        if not f or not sid:
            return {
                "error": "bad_request",
                "message": "file and spotify_user_id required",
            }, 400
        if f.mimetype.split("/")[0] != "image":
            return {"error": "unsupported_media_type"}, 415

        db = next(_get_db())
        acct = db.query(SpotifyAccount).filter_by(spotify_user_id=sid).one_or_none()
        if not acct:
            return {"error": "not_linked"}, 404

        try:
            acct = _refresh_if_needed(acct, db)
        except Exception as e:
            return {"error": "refresh_failed", "detail": str(e)}, 401

        # Ensure taste profile exists and is relatively fresh
        ut = _update_user_taste_if_stale(acct, db, max_age_hours=24)

        # OCR
        try:
            text_blob = _ocr_text_from_image(f.read())
        except ValueError as bad:
            return {"error": "bad_image", "message": str(bad)}, 400
        except RuntimeError as ocr:
            return {"error": "ocr_unavailable", "message": str(ocr)}, 503

        candidates = _tokenize_candidate_lines(text_blob)

        # Resolve each candidate on Spotify + score
        # Preload user's top artist names for co-occurrence scoring
        top_artists_long = _user_top_artists(acct, "long_term", limit=20)
        top_names = [a.get("name") for a in top_artists_long if a.get("name")]

        sess = _spotify_session()
        results: List[Dict[str, Any]] = []

        for cand in candidates:
            found = _spotify_search_artist(cand, acct.access_token)
            if not found:
                continue

            found_name = found["name"]
            name_score = fuzz.token_set_ratio(cand.lower(), found_name.lower()) / 100.0

            artist_genres = [g.lower() for g in found.get("genres", [])]
            genre_score = _jaccard(artist_genres, ut.genres or [])

            # optional co-occurrence bonus
            cooc = _cooccur_score(found_name, top_names, db)

            total = 0.55 * name_score + 0.30 * genre_score + 0.15 * cooc

            results.append(
                {
                    "candidate": cand,
                    "resolved_name": found_name,
                    "spotify_artist_id": found["id"],
                    "image": (found.get("images") or [{}])[0].get("url"),
                    "external_url": (found.get("external_urls") or {}).get("spotify"),
                    "genres": sorted(set(artist_genres))[:6],
                    "popularity": found.get("popularity"),
                    "scores": {
                        "name": round(name_score * 100, 1),
                        "genre": round(genre_score, 3),
                        "cooc": round(cooc, 3),
                        "total": round(total, 3),
                    },
                }
            )

        # Persist scan log + co-occurrence edges
        try:
            db.add(
                PosterScan(
                    spotify_user_id=sid,
                    scan_ts=int(time.time()),
                    artists_csv=",".join([r["resolved_name"] for r in results]),
                )
            )
            _log_cooccurrence([r["resolved_name"] for r in results], db)
        except Exception:
            # never fail the request if logging edges fails
            db.rollback()

        results.sort(key=lambda x: x["scores"]["total"], reverse=True)
        pruned = [r for r in results if r["scores"]["total"] >= 0.35][:20]
        return {"count": len(pruned), "items": pruned, "debug": {"candidates": candidates[:40]}}

    # ---------------------------
    # Error handlers
    # ---------------------------
    @app.errorhandler(413)
    def too_big(e):
        return {"error": "payload_too_large", "limit_mb": settings.max_upload_mb}, 413

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "not_found"}, 404

    @app.errorhandler(Exception)
    def internal(e):
        log.exception("Unhandled error")
        return {"error": "server_error"}, 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=settings.app_host, port=settings.app_port, debug=True)
