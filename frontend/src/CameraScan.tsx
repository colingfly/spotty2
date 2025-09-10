import { useEffect, useRef, useState } from "react";
import { API_BASE } from "./config";

type Props = {
  spotifyUserId: string;
  onResults: (artists: any[]) => void; // will pass back mapped Artist[]
};

export default function CameraScan({ spotifyUserId, onResults }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [starting, setStarting] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // start camera
  const startCamera = async () => {
    if (starting || stream) return;
    setError(null);
    setStarting(true);
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } }, // rear camera on mobile
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

  // stop camera
  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => {
    return () => stopCamera(); // cleanup on unmount
  }, []);

  const captureAndScan = async () => {
    if (!videoRef.current) return;
    setScanning(true);
    setError(null);
    try {
      const video = videoRef.current;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) {
        throw new Error("Camera not ready. Try again.");
      }

      // draw current frame to canvas
      if (!canvasRef.current) return;
      const canvas = canvasRef.current;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas not supported");
      ctx.drawImage(video, 0, 0, w, h);

      // convert to blob (JPEG)
      const blob: Blob = await new Promise((resolve) =>
        canvas.toBlob((b) => resolve(b as Blob), "image/jpeg", 0.92)
      );

      // send to backend
      const form = new FormData();
      form.append("spotify_user_id", spotifyUserId);
      form.append("file", blob, "capture.jpg");

      const res = await fetch(`${ API_BASE }/api/scan`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.message || "Scan failed");
      }

      // map results to your Artist shape (same mapping you use in App.tsx)
      const items = (data.items || []).map((x: any) => ({
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
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
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
        {/* hidden canvas used for capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>
        Tip: aim straight-on, fill the frame with the poster text. Avoid glare/blur.
      </div>
    </div>
  );
}
