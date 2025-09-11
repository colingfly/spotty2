# backend/app.py
from __future__ import annotations
import io, os, time, base64, logging
from urllib.parse import urlencode
from typing import Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import pytesseract
from rapidfuzz import fuzz

from sqlalchemy import text as sqltext

from config import settings
from db import SessionLocal
from models import SpotifyAccount

# ---------- Logging ----------
log = logging.getLogger("spotty")
logging.basicConfig(level=logging.INFO)

# ---------- HTTP client with retries ----------
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

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE  = "https://api.spotify.com/v1"

def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)  # Render/Heroku friendly
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024  # upload cap

    # CORS
    CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}}, supports_credentials=False)

    # Optional local Windows path for Tesseract
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    # ---- helpers ----
    def _basic_auth_header():
        raw = f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}

    def _get_db() -> Generator:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _refresh_if_needed(acct: SpotifyAccount, db) -> SpotifyAccount:
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
        db.add(acct); db.commit(); db.refresh(acct)
        return acct

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
            # Tesseract not available on host, or runtime issue
            raise RuntimeError("OCR unavailable on this server") from e

    # --- improved candidate extraction for venue calendars ---
    def _tokenize_candidate_lines(text: str) -> list[str]:
        raw = [ln.strip() for ln in text.splitlines()]

        STOP = {
            "get tickets","event date","event time","min ticket price","doors",
            "show","wed","thu","fri","sat","sun","mon","tue",
            "oct","nov","dec","pm","am","menu","calendar",
            "featured events","private events","about us","home"
        }

        keep=[]
        for ln in raw:
            if not ln:
                continue
            low = ln.lower()
            if any(s in low for s in STOP):
                continue
            clean = "".join(ch for ch in ln if ch.isalnum() or ch.isspace() or ch in "&-.'")
            clean = " ".join(clean.split())
            tokens = clean.split()
            if 1 <= len(tokens) <= 5 and any(c.isalpha() for c in clean):
                keep.append(clean)

        seen=set(); out=[]
        for k in keep:
            low=k.lower()
            if low not in seen:
                seen.add(low); out.append(k)
        return out

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _spotify_search_artist(name: str, bearer: str) -> dict | None:
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

    def _get_user_top_genres(acct: SpotifyAccount) -> set[str]:
        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/me/top/artists",
            headers={"Authorization": f"Bearer {acct.access_token}"},
            params={"time_range": "long_term", "limit": 50},
        )
        if r.status_code != 200:
            return set()
        artists = r.json().get("items", [])
        genres: set[str] = set()
        for a in artists:
            for g in a.get("genres", []):
                genres.add(g.lower())
        return genres

    # ====== Taste modeling helpers ======
    def _cosine(a, b):
        if not a or not b: return 0.0
        num = sum(x*y for x, y in zip(a, b))
        da  = (sum(x*x for x in a)) ** 0.5
        db  = (sum(y*y for y in b)) ** 0.5
        return (num / (da*db)) if da and db else 0.0

    def _genre_jaccard(A: set[str], B: set[str]) -> float:
        if not A or not B: return 0.0
        inter = len(A & B); union = len(A | B)
        return inter / union if union else 0.0

    def _audio_vec_mean(features_list: list[dict]) -> list[float]:
        if not features_list: return []
        vecs=[]
        for f in features_list:
            if not f: continue
            tempo = min(240.0, float(f.get("tempo", 0.0))) / 240.0
            vecs.append([
                float(f.get("danceability",0.0)),
                float(f.get("energy",0.0)),
                float(f.get("valence",0.0)),
                float(f.get("acousticness",0.0)),
                float(f.get("instrumentalness",0.0)),
                float(f.get("liveness",0.0)),
                float(f.get("speechiness",0.0)),
                tempo
            ])
        if not vecs: return []
        n=len(vecs)
        return [sum(col)/n for col in zip(*vecs)]

    def _artist_audio_vec(artist_id: str, sess: requests.Session, bearer: str) -> list[float]:
        r = sess.get(
            f"{API_BASE}/artists/{artist_id}/top-tracks",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"market": "US"},
        )
        if r.status_code != 200:
            return []
        tracks = r.json().get("tracks", [])[:10]
        if not tracks: return []
        track_ids=",".join(t["id"] for t in tracks if t.get("id"))
        r2 = sess.get(
            f"{API_BASE}/audio-features",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"ids": track_ids},
        )
        if r2.status_code != 200:
            return []
        feats = [f for f in r2.json().get("audio_features", []) if f]
        return _audio_vec_mean(feats)

    def _user_profile(acct: SpotifyAccount) -> dict:
        sess = _spotify_session()
        h  = {"Authorization": f"Bearer {acct.access_token}"}
        rt = sess.get(f"{API_BASE}/me/top/artists", headers=h, params={"time_range":"long_term", "limit": 25})
        if rt.status_code != 200:
            rt = sess.get(f"{API_BASE}/me/top/artists", headers=h, params={"time_range":"medium_term", "limit": 25})
        items = (rt.json() or {}).get("items", []) if rt.ok else []

        all_genres: set[str] = set()
        top_profiles = []
        for a in items:
            aid = a.get("id"); name = a.get("name"); genres = [g.lower() for g in a.get("genres",[])]
            for g in genres: all_genres.add(g)
            avec = _artist_audio_vec(aid, sess, acct.access_token) if aid else []
            top_profiles.append({"id": aid, "name": name, "genres": set(genres), "audio_vec": avec})

        vecs=[p["audio_vec"] for p in top_profiles if p["audio_vec"]]
        user_vec = [sum(col)/len(vecs) for col in zip(*vecs)] if vecs else []

        return {"audio_vec": user_vec, "genres": all_genres, "top_artists": top_profiles}

    def _sim_audio_genre(A_vec, A_genres, B_vec, B_genres):
        return 0.6 * _cosine(A_vec, B_vec) + 0.4 * _genre_jaccard(set(A_genres or []), set(B_genres or []))

    # --------- Routes ----------
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
            db.execute(sqltext("SELECT 1"))
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

        me = sess.get(f"{API_BASE}/me", headers={"Authorization": f"Bearer {access_token}"})
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
        db.add(acct); db.commit(); db.refresh(acct)

        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        return redirect(f"{frontend_url}/#sid={spotify_user_id}")

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
            return {"error": "spotify_error", "status_code": r.status_code, "body": r.text}, r.status_code

        items = (r.json() or {}).get("items", [])
        artists = [{
            "id": a.get("id"),
            "name": a.get("name"),
            "genres": a.get("genres", []),
            "popularity": a.get("popularity"),
            "image": (a.get("images") or [{}])[0].get("url"),
            "external_url": a.get("external_urls", {}).get("spotify"),
        } for a in items]
        return {"items": artists, "count": len(artists), "limit": limit, "time_range": time_range}

    @app.post("/api/scan")
    def api_scan():
        # Validate upload
        f = request.files.get("file")
        sid = request.form.get("spotify_user_id")
        if not f or not sid:
            return {"error": "bad_request", "message": "file and spotify_user_id required"}, 400
        if f.mimetype.split("/")[0] != "image":
            return {"error": "unsupported_media_type"}, 415

        # get user + refresh token if needed
        db = next(_get_db())
        acct = db.query(SpotifyAccount).filter_by(spotify_user_id=sid).one_or_none()
        if not acct:
            return {"error": "not_linked"}, 404
        try:
            acct = _refresh_if_needed(acct, db)
        except Exception as e:
            return {"error": "refresh_failed", "detail": str(e)}, 401

        try:
            text = _ocr_text_from_image(f.read())
        except ValueError as bad:
            return {"error": "bad_image", "message": str(bad)}, 400
        except RuntimeError as ocr:
            return {"error": "ocr_unavailable", "message": str(ocr)}, 503

        candidates = _tokenize_candidate_lines(text)
        user_genres = _get_user_top_genres(acct)

        # first pass: resolve names to Spotify & compute name/genre confidence
        sess = _spotify_session()
        base_results = []
        for cand in candidates:
            found = _spotify_search_artist(cand, acct.access_token)
            if not found:
                continue
            found_name = found["name"]
            name_score = fuzz.token_set_ratio(cand.lower(), found_name.lower())  # 0..100
            artist_genres = set(g.lower() for g in found.get("genres", []))
            genre_score = _jaccard(artist_genres, user_genres)                  # 0..1
            total = 0.6 * (name_score / 100.0) + 0.4 * genre_score
            base_results.append({
                "candidate": cand,
                "resolved_name": found_name,
                "spotify_artist_id": found["id"],
                "image": (found.get("images") or [{}])[0].get("url"),
                "external_url": (found.get("external_urls") or {}).get("spotify"),
                "genres": sorted(artist_genres)[:6],
                "popularity": found.get("popularity"),
                "scores": {
                    "name": round(name_score, 1),
                    "genre": round(genre_score, 3),
                    "total": round(total, 3)
                }
            })

        # user taste profile
        profile = _user_profile(acct)
        user_vec  = profile["audio_vec"]
        user_gset = profile["genres"]
        user_tops = profile["top_artists"]

        enriched=[]
        for r in base_results:
            aid = r.get("spotify_artist_id")
            a_genres = set(r.get("genres", []))
            a_vec = _artist_audio_vec(aid, sess, acct.access_token) if aid else []
            sim_to_user = _sim_audio_genre(a_vec, a_genres, user_vec, user_gset) if a_vec and user_vec else 0.0

            closest_name = None; closest_sim = 0.0
            for up in user_tops:
                if not up["audio_vec"]:
                    continue
                s = _sim_audio_genre(a_vec, a_genres, up["audio_vec"], up["genres"])
                if s > closest_sim:
                    closest_sim = s
                    closest_name = up["name"]

            name_score  = (r.get("scores", {}) or {}).get("name", 0.0) / 100.0
            genre_score = (r.get("scores", {}) or {}).get("genre", 0.0)
            confidence  = 0.6*name_score + 0.4*genre_score

            final = 0.6 * sim_to_user + 0.4 * confidence

            r["taste"] = {
                "sim_to_user": round(sim_to_user,3),
                "closest_user_artist": closest_name,
                "closest_user_sim": round(closest_sim,3),
                "final_score": round(final,3),
                "why": [w for w in [
                    f"closest to your '{closest_name}'" if closest_name and closest_sim>=0.45 else None,
                    "strong text/genre match" if confidence>=0.5 else None,
                ] if w]
            }
            enriched.append(r)

        enriched.sort(key=lambda x: x.get("taste",{}).get("final_score",0.0), reverse=True)
        pruned = [r for r in enriched if r.get("taste",{}).get("final_score",0.0) >= 0.35][:25]

        return {
            "count": len(pruned),
            "items": pruned,
            "debug": {"candidates": candidates[:40], "used_user_genres": sorted(list(user_gset))[:20]}
        }

    # ---- Error handlers ----
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
