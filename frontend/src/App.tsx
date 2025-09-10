import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config";
import CameraScan from "./CameraScan";

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
  const [timeRange, setTimeRange] =
    useState<"short_term" | "medium_term" | "long_term">("long_term");
  const [artists, setArtists] = useState<Artist[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [showCamera, setShowCamera] = useState(false);

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

  const scanImage = async () => {
    if (!sid) { alert("Link Spotify first"); return; }
    if (!file) { alert("Choose an image"); return; }

    const form = new FormData();
    form.append("spotify_user_id", sid);
    form.append("file", file);

    const res = await fetch(`${API_BASE}/api/scan`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      alert(data?.message || "Scan failed");
      return;
    }

    const items: Artist[] = (data.items || []).map((x: any) => ({
      id: x.spotify_artist_id,
      name: x.resolved_name,
      genres: x.genres || [],
      image: x.image,
      external_url: x.external_url,
      popularity: x.popularity ?? undefined,                    // Spotify popularity
      matchTotal: Math.round((x.scores?.total ?? 0) * 100),     // 0..100
      nameSim: Math.round(x.scores?.name ?? 0),                 // 0..100
      genreSim: Math.round((x.scores?.genre ?? 0) * 100),       // 0..100
      fromScan: true,
    }));
    items.sort((a, b) => (b.matchTotal ?? 0) - (a.matchTotal ?? 0));
    setArtists(items);
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

        {/* Quick mobile camera picker */}
        <input
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button onClick={scanImage} disabled={!sid || !file}>
          Scan image
        </button>

        {/* Live camera */}
        <button onClick={() => setShowCamera((s) => !s)}>
          {showCamera ? "Hide camera" : "Use camera"}
        </button>
      </div>

      {showCamera && sid && (
        <div style={{ marginBottom: 16 }}>
          <CameraScan
            spotifyUserId={sid}
            onResults={(items) => setArtists(items)}
          />
        </div>
      )}

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
