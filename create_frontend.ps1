# Creates a mobile-friendly Vite + React + TS frontend with camera capture for Spotty
# Run from the spotty2 directory (where /backend lives)

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$FRONT = Join-Path $ROOT "frontend"
$SRC   = Join-Path $FRONT "src"
$PUB   = Join-Path $FRONT "public"

New-Item -ItemType Directory -Force -Path $FRONT,$SRC,$PUB | Out-Null

# ---------------- package.json ----------------
@'
{
  "name": "spotty-mobile-frontend",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --host --https"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.4",
    "vite": "^5.4.0"
  }
}
'@ | Set-Content (Join-Path $FRONT "package.json") -Encoding UTF8

# ---------------- tsconfig.json ----------------
@'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "jsx": "react-jsx",
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true
  },
  "include": ["src"]
}
'@ | Set-Content (Join-Path $FRONT "tsconfig.json") -Encoding UTF8

# ---------------- vite.config.ts ----------------
@'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,   // expose on LAN
    https: true,  // self-signed for mobile camera
    port: 5173
  }
});
'@ | Set-Content (Join-Path $FRONT "vite.config.ts") -Encoding UTF8

# ---------------- index.html ----------------
@'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1, viewport-fit=cover"
    />
    <link rel="manifest" href="/manifest.webmanifest" />
    <meta name="theme-color" content="#111111" />
    <title>Spotty Mobile</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
'@ | Set-Content (Join-Path $FRONT "index.html") -Encoding UTF8

# ---------------- public/manifest.webmanifest ----------------
@'
{
  "name": "Spotty Mobile",
  "short_name": "Spotty",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#111111",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
'@ | Set-Content (Join-Path $PUB "manifest.webmanifest") -Encoding UTF8

# ---------------- placeholder icons (1x1 PNG via base64) ----------------
$pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
[IO.File]::WriteAllBytes((Join-Path $PUB "icon-192.png"), [Convert]::FromBase64String($pngBase64))
[IO.File]::WriteAllBytes((Join-Path $PUB "icon-512.png"), [Convert]::FromBase64String($pngBase64))

# ---------------- src/config.ts ----------------
@'
export const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8888";
'@ | Set-Content (Join-Path $SRC "config.ts") -Encoding UTF8

# ---------------- src/mobile.css ----------------
@'
:root { --pad: 12px; --radius: 12px; }
main { padding: var(--pad); }
h1 { font-size: clamp(28px, 6vw, 56px); line-height: 1.15; }
button, select, input[type="file"] {
  font-size: 16px; /* prevents iOS zoom */
  padding: 10px 14px;
  border-radius: var(--radius);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: var(--pad);
}
.card {
  border: 1px solid #e5e5e5;
  border-radius: var(--radius);
  padding: var(--pad);
  background: #fff;
}
@media (max-width: 480px) {
  .grid { grid-template-columns: 1fr 1fr; }
  .card img { height: 160px; }
}
.stickybar {
  position: sticky; top: 0; z-index: 10; background: #fff;
  padding: 8px 0; border-bottom: 1px solid #eee;
}
'@ | Set-Content (Join-Path $SRC "mobile.css") -Encoding UTF8

# ---------------- src/main.tsx ----------------
@'
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./mobile.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
'@ | Set-Content (Join-Path $SRC "main.tsx") -Encoding UTF8

# ---------------- src/CameraScan.tsx ----------------
@'
import { useEffect, useRef, useState } from "react";
import { BACKEND_URL } from "./config";

type Props = {
  spotifyUserId: string;
  onResults: (artists: any[]) => void;
};

export default function CameraScan({ spotifyUserId, onResults }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [starting, setStarting] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = async () => {
    if (starting || stream) return;
    setError(null);
    setStarting(true);
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      setStream(media);
      if (videoRef.current) {
        videoRef.current.srcObject = media;
        await videoRef.current.play();
      }
    } catch (e: any) {
      setError(e.message || "Camera access failed. Use file upload instead.");
    } finally {
      setStarting(false);
    }
  };

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  useEffect(() => {
    return () => stopCamera();
  }, []);

  const captureAndScan = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    setScanning(true);
    setError(null);
    try {
      const video = videoRef.current;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) throw new Error("Camera not ready. Try again.");

      const canvas = canvasRef.current;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = w * Math.min(dpr, 2);
      canvas.height = h * Math.min(dpr, 2);
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas not supported");
      ctx.scale(canvas.width / w, canvas.height / h);
      ctx.drawImage(video, 0, 0, w, h);

      const quality = window.innerWidth < 600 ? 0.8 : 0.92;
      const blob: Blob = await new Promise((resolve) =>
        canvas.toBlob((b) => resolve(b as Blob), "image/jpeg", quality)
      );

      const form = new FormData();
      form.append("spotify_user_id", spotifyUserId);
      form.append("file", blob, "capture.jpg");

      const res = await fetch(`${BACKEND_URL}/api/scan`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Scan failed");

      const items = (data.items || []).map((x: any) => ({
        id: x.spotify_artist_id,
        name: x.resolved_name,
        genres: x.genres || [],
        image: x.image,
        external_url: x.external_url,
        popularity: x.popularity ?? undefined,
        matchTotal: Math.round((x.scores?.total ?? 0) * 100),
        nameSim: Math.round(x.scores?.name ?? 0),
        genreSim: Math.round((x.scores?.genre ?? 0) * 100),
        fromScan: true,
      }));
      items.sort((a: any, b: any) => (b.matchTotal ?? 0) - (a.matchTotal ?? 0));
      onResults(items);
    } catch (e: any) {
      setError(e.message || "Capture failed");
    } finally {
      setScanning(false);
    }
  };

  return (
    <div style={{ border: "1px solid #eee", padding: 12, borderRadius: 12 }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        {!stream ? (
          <button onClick={startCamera} disabled={starting}>
            {starting ? "Starting camera…" : "Start camera"}
          </button>
        ) : (
          <>
            <button onClick={captureAndScan} disabled={scanning}>
              {scanning ? "Scanning…" : "Capture & Scan"}
            </button>
            <button onClick={stopCamera}>Stop camera</button>
          </>
        )}
      </div>

      {error && <div style={{ color: "crimson", marginBottom: 8 }}>{error}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ width: "100%", borderRadius: 8, background: "#000" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>
        Tip: aim straight-on, fill the frame with the poster text. Avoid glare/blur.
      </div>
    </div>
  );
}
'@ | Set-Content (Join-Path $SRC "CameraScan.tsx") -Encoding UTF8

# ---------------- src/App.tsx ----------------
@'
import { useEffect, useMemo, useState } from "react";
import { BACKEND_URL } from "./config";
import CameraScan from "./CameraScan";

type Artist = {
  id: string;
  name: string;
  genres: string[];
  url?: string;
  external_url?: string;
  image?: string | null;
  popularity?: number;
  matchTotal?: number;
  nameSim?: number;
  genreSim?: number;
  fromScan?: boolean;
};

function useSpotifyUserId() {
  const [sid, setSid] = useState<string | null>(null);
  useEffect(() => {
    const hash = window.location.hash;
    const m = /sid=([^&]+)/.exec(hash);
    if (m?.[1]) {
      localStorage.setItem("spotify_user_id", m[1]);
      setSid(m[1]);
      history.replaceState(null, "", window.location.pathname);
      return;
    }
    const saved = localStorage.getItem("spotify_user_id");
    if (saved) setSid(saved);
  }, []);
  return sid;
}

export default function App() {
  const sid = useSpotifyUserId();
  const [timeRange, setTimeRange] = useState<"short_term" | "medium_term" | "long_term">("long_term");
  const [artists, setArtists] = useState<Artist[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [showCamera, setShowCamera] = useState(false);

  const loginUrl = useMemo(() => `${BACKEND_URL}/login`, []);

  const fetchArtists = async () => {
    if (!sid) { setError("Not linked yet. Click Connect Spotify first."); return; }
    setLoading(true); setError(null);
    const url = `${BACKEND_URL}/api/me/top-artists?spotify_user_id=${encodeURIComponent(sid)}&time_range=${timeRange}&limit=24`;
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setArtists(data.items || []);
    } catch (e: any) {
      setError(e.message || "Failed to load artists"); setArtists([]);
    } finally { setLoading(false); }
  };

  const scanImage = async () => {
    if (!sid) { alert("Link Spotify first"); return; }
    if (!file) { alert("Choose an image"); return; }
    const form = new FormData();
    form.append("spotify_user_id", sid);
    form.append("file", file);
    const res = await fetch(`${BACKEND_URL}/api/scan`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) { alert(data?.message || "Scan failed"); return; }
    const items: Artist[] = (data.items || []).map((x: any) => ({
      id: x.spotify_artist_id,
      name: x.resolved_name,
      genres: x.genres || [],
      image: x.image,
      external_url: x.external_url,
      popularity: x.popularity ?? undefined,
      matchTotal: Math.round((x.scores?.total ?? 0) * 100),
      nameSim: Math.round(x.scores?.name ?? 0),
      genreSim: Math.round((x.scores?.genre ?? 0) * 100),
      fromScan: true,
    }));
    items.sort((a, b) => (b.matchTotal ?? 0) - (a.matchTotal ?? 0));
    setArtists(items);
  };

  return (
    <main>
      <div className="stickybar">
        <h1 style={{ margin: 0, padding: "0 8px" }}>Spotty Mobile</h1>
      </div>

      {!sid ? (
        <a href={loginUrl}>
          <button style={{ marginTop: 12 }}>Connect Spotify</button>
        </a>
      ) : (
        <div style={{ margin: "12px 0", display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <div>Linked as: <code>{sid}</code></div>
          <button onClick={() => { localStorage.removeItem("spotify_user_id"); location.reload(); }}>
            Unlink
          </button>
        </div>
      )}

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <label>Time range:</label>
        <select value={timeRange} onChange={(e) => setTimeRange(e.target.value as any)}>
          <option value="short_term">Last 4 weeks</option>
          <option value="medium_term">Last 6 months</option>
          <option value="long_term">All time</option>
        </select>

        <button onClick={fetchArtists} disabled={!sid || loading}>
          {loading ? "Loading..." : "Load Top Artists"}
        </button>

        <input type="file" accept="image/*" capture="environment" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={scanImage} disabled={!sid || !file}>Scan image</button>

        <button onClick={() => setShowCamera((s) => !s)}>
          {showCamera ? "Hide camera" : "Use camera"}
        </button>
      </div>

      {showCamera && sid && (
        <div style={{ marginBottom: 12 }}>
          <CameraScan spotifyUserId={sid} onResults={(items) => setArtists(items)} />
        </div>
      )}

      {error && <div style={{ color: "crimson", marginBottom: 12 }}>{error}</div>}
      {!error && !loading && artists.length === 0 && (
        <div style={{ color: "#666", marginBottom: 12 }}>
          No artists yet. Load Top Artists or scan a poster.
        </div>
      )}

      <div className="grid">
        {artists.map((a) => (
          <div key={a.id + (a.fromScan ? "-scan" : "-top")} className="card">
            <img
              src={a.image ?? "https://via.placeholder.com/300x300?text=No+Image"}
              alt={a.name}
              style={{ width: "100%", height: 220, objectFit: "cover", borderRadius: 8 }}
            />
            <h3 style={{ margin: "10px 0 4px" }}>{a.name}</h3>
            <div style={{ fontSize: 12, color: "#666", minHeight: 32 }}>
              {a.genres?.slice(0, 3).join(", ")}
            </div>

            {a.fromScan ? (
              <div style={{ fontSize: 12, color: "#333", lineHeight: 1.5 }}>
                <div><strong>Total match:</strong> {a.matchTotal ?? "—"}%</div>
                <div><strong>Name similarity:</strong> {a.nameSim ?? "—"}%</div>
                <div><strong>Genre similarity:</strong> {a.genreSim ?? "—"}%</div>
                {typeof a.popularity === "number" && (
                  <div><strong>Popularity (Spotify):</strong> {a.popularity}</div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: "#333" }}>
                Popularity (Spotify): {a.popularity ?? "—"}
              </div>
            )}

            <div style={{ marginTop: 8 }}>
              <a href={(a as any).external_url || a.url || "#"} target="_blank" rel="noreferrer">
                Open in Spotify
              </a>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
'@ | Set-Content (Join-Path $SRC "App.tsx") -Encoding UTF8

Write-Host "✅ Frontend scaffolded in $FRONT"
Write-Host "Next:"
Write-Host "  cd frontend"
Write-Host "  npm i"
Write-Host "  npm run dev"
Write-Host "Then open https://<your-LAN-ip>:5173 on your phone (accept the self-signed cert)."
