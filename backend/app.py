# backend/app.py
from __future__ import annotations
import io, os, time, base64, logging, math
from urllib.parse import urlencode
from typing import Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import Flask, request, redirect
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import pytesseract
from rapidfuzz import fuzz
from sqlalchemy import text as sql_text

from config import settings
from db import SessionLocal
from models import SpotifyAccount, UserTaste, PosterScan, ArtistEdge


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
    return s


AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE  = "https://api.spotify.com/v1"

AUDIO_KEYS = [
    "danceability","energy","valence","acousticness",
    "instrumentalness","liveness","speechiness","tempo"
]


def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)  # proxy friendly (Render/Heroku)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024  # upload cap
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
            timeout=15
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

    def _tokenize_candidate_lines(text: str) -> list[str]:
        raw = [ln.strip() for ln in text.splitlines()]
        keep: list[str] = []
        for ln in raw:
            if not ln or len(ln) < 2:
                continue
            if any(ch.isdigit() for ch in ln) and len(ln) <= 4:
                continue
            clean = "".join(ch for ch in ln if ch.isalnum() or ch.isspace() or ch in "&-.'")
            words = clean.split()
            if len(words) > 7:
                continue
            keep.append(clean.strip())
        # de-dupe (case-insensitive)
        seen = set(); out=[]
        for k in keep:
            low = k.lower()
            if low not in seen:
                seen.add(low); out.append(k)
        return out

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # ---------- music-expert helpers ----------
    def _get_user_taste(acct: SpotifyAccount, db) -> dict:
        """
        Cache & return user's taste vector and genre set.
        Vector = average of top tracks' audio features (long_term).
        """
        now = int(time.time())
        ut = db.query(UserTaste).filter_by(spotify_user_id=acct.spotify_user_id).one_or_none()
        # refresh every 24h
        if not ut or now - (ut.updated_at or 0) > 86400:
            sess = _spotify_session()
            r = sess.get(
                f"{API_BASE}/me/top/tracks",
                headers={"Authorization": f"Bearer {acct.access_token}"},
                params={"time_range": "long_term", "limit": 50},
                timeout=15,
            )
            tracks = r.json().get("items", []) if r.ok else []
            ids = [t["id"] for t in tracks if t.get("id")]

            af = {}
            if ids:
                rr = sess.get(
                    f"{API_BASE}/audio-features",
                    headers={"Authorization": f"Bearer {acct.access_token}"},
                    params={"ids": ",".join(ids[:100])},
                    timeout=15,
                )
                if rr.ok:
                    af = {x["id"]: x for x in rr.json().get("audio_features", []) if x}

            vec = {k: 0.0 for k in AUDIO_KEYS}
            n = 0
            for tid in ids:
                f = af.get(tid)
                if not f: continue
                for k in AUDIO_KEYS:
                    vec[k] += float(f.get(k, 0.0))
                n += 1
            if n:
                for k in AUDIO_KEYS: vec[k] /= n

            # union genres from artists on those tracks
            genres = set()
            for t in tracks:
                for a in t.get("artists", []):
                    ar = sess.get(
                        f"{API_BASE}/artists/{a['id']}",
                        headers={"Authorization": f"Bearer {acct.access_token}"},
                        timeout=15,
                    )
                    if ar.ok:
                        for g in (ar.json().get("genres") or []):
                            genres.add(g.lower())

            if not ut:
                ut = UserTaste(spotify_user_id=acct.spotify_user_id)
            ut.updated_at = now
            for k,v in vec.items(): setattr(ut, k, v)
            ut.genres = ",".join(sorted(genres))
            db.add(ut); db.commit(); db.refresh(ut)

        vec = {k: getattr(ut, k) or 0.0 for k in AUDIO_KEYS}
        genres = set((ut.genres or "").split(",")) if ut.genres else set()
        return {"vec": vec, "genres": genres}

    def _get_artist_audio_vec(artist_spotify_id: str, bearer: str) -> dict | None:
        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/artists/{artist_spotify_id}/top-tracks",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"market": "US"},
            timeout=15,
        )
        if not r.ok:
            return None
        tracks = r.json().get("tracks", [])
        ids = [t["id"] for t in tracks if t.get("id")]
        if not ids:
            return None

        rr = sess.get(
            f"{API_BASE}/audio-features",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"ids": ",".join(ids[:100])},
            timeout=15,
        )
        if not rr.ok:
            return None
        feats = rr.json().get("audio_features", [])
        if not feats:
            return None

        vec = {k: 0.0 for k in AUDIO_KEYS}
        n = 0
        for f in feats:
            if not f: continue
            for k in AUDIO_KEYS:
                vec[k] += float(f.get(k, 0.0))
            n += 1
        if n:
            for k in AUDIO_KEYS: vec[k] /= n
        return vec

    def _cosine(a: dict, b: dict) -> float:
        num = 0.0; da = 0.0; db = 0.0
        for k in AUDIO_KEYS:
            x = float(a.get(k, 0.0)); y = float(b.get(k, 0.0))
            num += x * y; da += x * x; db += y * y
        if da <= 0 or db <= 0: return 0.0
        return num / math.sqrt(da * db)

    def _spotify_search_artist(name: str, bearer: str) -> dict | None:
        sess = _spotify_session()
        r = sess.get(
            f"{API_BASE}/search",
            headers={"Authorization": f"Bearer {bearer}"},
            params={"q": name, "type": "artist", "limit": 3},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        items = (r.json().get("artists") or {}).get("items") or []
        return items[0] if items else None

    # --- co-occurrence helpers ---
    def _stable_pair(a: str, b: str) -> tuple[str, str]:
        a = a.lower().strip(); b = b.lower().strip()
        return (a, b) if a <= b else (b, a)

    def _update_cooccurrence(db, resolved_names: list[str]):
        now = int(time.time())
        names = [n.lower().strip() for n in resolved_names if n]
        names = list(dict.fromkeys(names))  # unique
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = _stable_pair(names[i], names[j])
                edge = db.query(ArtistEdge).filter_by(a=a, b=b).one_or_none()
                if not edge:
                    edge = ArtistEdge(a=a, b=b, weight=0.0, last_seen=now)
                edge.weight += 1.0
                edge.last_seen = now
                db.add(edge)
        db.commit()

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
            db.execute(sql_text("SELECT 1"))
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
            timeout=15,
        )
        if r.status_code != 200:
            return {"error": "token_exchange_failed", "body": r.text}, r.status_code
        tokens = r.json()
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")
        expires_at = int(time.time()) + int(tokens.get("expires_in", 3600))

        me = sess.get(f"{API_BASE}/me", headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
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
            timeout=20,
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
        f = request.files.get("file")
        sid = request.form.get("spotify_user_id")
        if not f or not sid:
            return {"error": "bad_request", "message": "file and spotify_user_id required"}, 400
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

        try:
            text = _ocr_text_from_image(f.read())
        except ValueError as bad:
            return {"error": "bad_image", "message": str(bad)}, 400
        except RuntimeError as ocr:
            return {"error": "ocr_unavailable", "message": str(ocr)}, 503

        candidates = _tokenize_candidate_lines(text)

        # --- build "music-expert" scoring ---
        user = _get_user_taste(acct, db)
        user_vec = user["vec"]
        user_genres = user["genres"]

        results = []
        sess = _spotify_session()

        for cand in candidates:
            found = _spotify_search_artist(cand, acct.access_token)
            if not found:
                continue

            found_name = found["name"]; artist_id = found["id"]

            # name (0..1)
            name_score = fuzz.token_set_ratio(cand.lower(), found_name.lower()) / 100.0

            # genre (0..1)
            artist_genres = set(g.lower() for g in found.get("genres", []))
            genre_score = _jaccard(artist_genres, user_genres)

            # audio cosine (0..1)
            avec = _get_artist_audio_vec(artist_id, acct.access_token) or {}
            audio_score = _cosine(user_vec, avec)

            # popularity prior (0..1)
            pop = found.get("popularity", 0)
            pop_score = float(pop) / 100.0

            # co-occurrence boost (0..0.15)
            boost = 0.0
            try:
                top_seen = db.query(ArtistEdge).filter(
                    (ArtistEdge.a == found_name.lower()) | (ArtistEdge.b == found_name.lower())
                ).order_by(ArtistEdge.weight.desc()).limit(1).one_or_none()
                maxw = top_seen.weight if top_seen else 0.0
                boost = min(maxw / 10.0, 0.15)
            except Exception:
                boost = 0.0

            total = (
                0.25 * name_score +
                0.30 * genre_score +
                0.35 * audio_score +
                0.10 * pop_score +
                boost
            )

            reason_bits = []
            if name_score >= 0.7: reason_bits.append("name match")
            if genre_score >= 0.3: reason_bits.append("genre overlap")
            if audio_score >= 0.35: reason_bits.append("similar audio feel")
            if pop_score >= 0.5: reason_bits.append("popular pick")
            if boost > 0.0: reason_bits.append("seen together on posters")

            results.append({
                "candidate": cand,
                "resolved_name": found_name,
                "spotify_artist_id": artist_id,
                "image": (found.get("images") or [{}])[0].get("url"),
                "external_url": (found.get("external_urls") or {}).get("spotify"),
                "genres": sorted(artist_genres)[:4],
                "popularity": found.get("popularity"),
                "scores": {
                    "name": round(100 * name_score, 1),
                    "genre": round(100 * genre_score, 1),
                    "audio": round(100 * audio_score, 1),
                    "pop": round(100 * pop_score, 1),
                    "total": round(total, 3),
                },
                "explanation": " + ".join(reason_bits) if reason_bits else "closest overall match",
            })

        # co-occurrence update + scan log
        resolved_names = [r["resolved_name"] for r in results]
        if resolved_names:
            _update_cooccurrence(db, resolved_names)
            db.add(PosterScan(
                spotify_user_id=sid,
                scan_ts=int(time.time()),
                artists_csv=",".join(resolved_names)
            ))
            db.commit()

        # rank & prune
        results.sort(key=lambda x: x["scores"]["total"], reverse=True)
        pruned = [r for r in results if r["scores"]["total"] >= 0.35][:20]
        return {"count": len(pruned), "items": pruned, "debug": {"candidates": candidates[:30]}}

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
