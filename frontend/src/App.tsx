import React, { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config";

type Artist = {
  id: string;
  name: string;
  genres: string[];
  url?: string;
  external_url?: string;
  image?: string | null;
  popularity?: number;

  // Scan-only fields
  matchTotal?: number; // 0..100
  nameSim?: number;    // 0..100
  genreSim?: number;   // 0..100
  fromScan?: boolean;
};

function useSpotifyUserId() {
  const [sid, setSid] = useState<string | null>(null);

  useEffect(() => {
    // take sid from hash once after login redirect
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

// Optional: compress images client-side before upload
async function compressToJpeg(file: File, maxDim = 1600, quality = 0.9): Promise<Blob> {
  // If it's not an image, just return the original file
  if (!file.type.startsWith("image/")) return file;

  const img = await createImageBitmap(file);
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported");
  ctx.drawImage(img, 0, 0, w, h);

  const blob: Blob = await new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/jpeg", quality)
  );
  return blob;
}

export default function App() {
  const sid = useSpotifyUserId();

  const [timeRange, setTimeRange] =
    useState<"short_term" | "medium_term" | "long_term">("long_term");
  const [artists, setArtists] = useState<Artist[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selFile, setSelFile] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);

  const loginUrl = useMemo(() => `${API_BASE}/login`, []);

  const fetchArtists = async () => {
    if (!sid) {
      setError("Not linked yet. Click Connect Spotify first.");
      return;
    }
    setLoading(true);
    setError(null);

    const url = `${API_BASE}/api/me/top-artists?spotify_user_id=${encodeURIComponent(
      sid
    )}&time_range=${timeRange}&limit=24`;

    try {
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setArtists(data.items || []);
    } catch (e: any) {
      setError(e.message || "Failed to load artists");
      setArtists([]);
    } finally {
      setLoading(false);
    }
  };

  const handleScan = async () => {
    try {
      if (!sid) {
        alert("Connect Spotify first.");
        return;
      }
      if (!selFile) {
        alert("Please select an image.");
        return;
      }

      setScanning(true);

      // Optional compression to speed uploads / avoid 413
      const payload = await compressToJpeg(selFile, 1600, 0.9);

      const form = new FormData();
      form.append("spotify_user_id", sid);
      form.append("file", payload, "poster.jpg");

      const res = await fetch(`${API_BASE}/api/scan`, { method: "POST", body: form });
      let data: any = null;
      try { data = await res.json(); } catch { /* ignore */ }

      if (!res.ok) {
        const msg =
          data?.message ||
          (data?.error === "ocr_unavailable"
            ? "OCR is unavailable on the server."
            : `Scan failed (HTTP ${res.status})`);
        alert(msg);
        console.error("scan error:", res.status, data);
        return;
      }

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

      // reset file picker for convenience
      setSelFile(null);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || "Scan failed.");
    } finally {
      setScanning(false);
    }
  };

  return (
    <main style={{ maxWidth: 1000, margin: "40px auto", padding: 16 }}>
      <h1 style={{ fontSize: 56, marginBottom: 8 }}>Spotify Top Artists</h1>

      {!sid ? (
        <a href={loginUrl}>
          <button style={{ fontSize: 18, padding: "10px 16px" }}>Connect Spotify</button>
        </a>
      ) : (
        <div style={{ marginBottom: 16, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ fontSize: 20 }}>
            Linked as: <code>{sid}</code>
          </div>
          <button
            onClick={() => {
              localStorage.removeItem("spotify_user_id");
              location.reload();
            }}
          >
            Unlink
          </button>
        </div>
      )}

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <label>Time range:</label>
        <select value={timeRange} onChange={(e) => setTimeRange(e.target.value as any)}>
          <option value="short_term">Last 4 weeks</option>
          <option value="medium_term">Last 6 months</option>
          <option value="long_term">All time</option>
        </select>

        <button onClick={fetchArtists} disabled={!sid || loading} style={{ padding: "8px 12px" }}>
          {loading ? "Loading..." : "Load Top Artists"}
        </button>

        {/* Upload-only scan: on mobile, this opens the camera */}
        <input
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(e) => setSelFile(e.target.files?.[0] ?? null)}
        />
        <button
          onClick={handleScan}
          disabled={!sid || !selFile || scanning}
          title={!sid ? "Connect Spotify first" : (!selFile ? "Select an image" : "")}
        >
          {scanning ? "Scanning…" : "Scan image"}
        </button>
      </div>

      {error && <div style={{ color: "crimson", marginBottom: 12 }}>{error}</div>}
      {!error && !loading && artists.length === 0 && (
        <div style={{ color: "#666", marginBottom: 12 }}>
          No artists yet. Load Top Artists or scan a poster.
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
          gap: 16,
        }}
      >
        {artists.map((a) => (
          <div
            key={a.id + (a.fromScan ? "-scan" : "-top")}
            style={{
              border: "1px solid #e5e5e5",
              borderRadius: 12,
              padding: 12,
              background: "white",
            }}
          >
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
              <a
                href={(a as any).external_url || a.url || "#"}
                target="_blank"
                rel="noreferrer"
              >
                Open in Spotify
              </a>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
