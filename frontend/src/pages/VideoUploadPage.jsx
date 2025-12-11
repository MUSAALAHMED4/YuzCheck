import React, { useState, useEffect, useRef } from "react";
import { uploadVideo, getLatestRecognition } from "../api/attendanceApi";
import LatestRecognitionCard from "../components/LatestRecognitionCard";
import AttendanceTable from "../components/AttendanceTable";
import AbsenceTable from "../components/AbsenceTable";

export default function VideoUploadPage() {
  const [file, setFile] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [latestRecognition, setLatestRecognition] = useState(null);
  const [recognized, setRecognized] = useState([]);
  const [absent, setAbsent] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const videoRef = useRef(null);
  const [progress, setProgress] = useState({ current: 0, duration: 0, pct: 0 });

  const handleUpload = async (f) => {
    const selected = f || file;
    if (!selected) return;
    setLoading(true);
    setIsProcessing(true);
    try {
      const info = await uploadVideo(selected);
      setVideoInfo(info);
      // Try to populate results if backend returns them
      const r = info?.result?.recognized || [];
      const a = info?.result?.absent || [];
      if (Array.isArray(r)) setRecognized(r);
      if (Array.isArray(a)) setAbsent(a);
    } finally {
      setLoading(false);
      // Keep isProcessing true while backend might still be recognizing
    }
  };

  // Start/stop polling only while processing is active
  useEffect(() => {
    if (!isProcessing) return;
    const interval = setInterval(async () => {
      try {
        const latest = await getLatestRecognition();
        if (latest && latest.name) setLatestRecognition(latest);
      } catch (err) {
        console.error("Failed to fetch latest recognition:", err);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isProcessing]);

  // Handle preview when file changes
  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      setIsProcessing(false);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    // autoplay preview
    setTimeout(() => {
      try {
        if (videoRef.current) {
          videoRef.current.src = url;
          videoRef.current.play().catch(() => {});
        }
      } catch {}
    }, 0);
    // Auto start upload/recognition when a video is selected
    handleUpload(file);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div style={{ display: "flex", gap: "16px" }}>
      {/* Main Content */}
      <div style={{ flex: 1 }}>
        <div className="card" style={{ maxWidth: 560 }}>
          <h2>أخذ الحضور من فيديو</h2>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            style={{ margin: "12px 0" }}
          />
          {/* Video Preview Container (like live page) */}
          {previewUrl && (
            <div style={{ marginTop: 12 }}>
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  maxWidth: 600,
                  height: 320,
                  borderRadius: 18,
                  overflow: "hidden",
                  border: "2.5px solid #2563eb",
                  boxShadow: "0 6px 32px 0 rgba(37,99,235,0.10)",
                  background:
                    "linear-gradient(135deg, #1e293b 0%, #2563eb 100%)",
                  margin: "0 auto",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <video
                  ref={videoRef}
                  controls
                  onLoadedMetadata={(e) => {
                    try {
                      e.currentTarget.currentTime = 0;
                      const d = e.currentTarget.duration || 0;
                      setProgress({ current: 0, duration: d, pct: 0 });
                      e.currentTarget.play().catch(() => {});
                    } catch {}
                  }}
                  onEnded={() => setIsProcessing(false)}
                  onTimeUpdate={(e) => {
                    const cur = e.currentTarget.currentTime || 0;
                    const dur = e.currentTarget.duration || 0;
                    const pct =
                      dur > 0
                        ? Math.min(100, Math.max(0, (cur / dur) * 100))
                        : 0;
                    setProgress({ current: cur, duration: dur, pct });
                  }}
                  style={{
                    width: "98%",
                    height: "96%",
                    objectFit: "cover",
                    background: "#0b1220",
                    borderRadius: "14px",
                    boxShadow: "0 2px 12px rgba(0,0,0,0.10)",
                  }}
                />
                {/* Progress bar */}
                {progress.duration > 0 && (
                  <div
                    style={{
                      position: "absolute",
                      bottom: 42,
                      left: 12,
                      right: 12,
                    }}
                  >
                    <div
                      style={{
                        height: 8,
                        background: "#0b1220",
                        borderRadius: 999,
                        boxShadow: "inset 0 0 6px rgba(0,0,0,0.3)",
                      }}
                    >
                      <div
                        style={{
                          width: `${progress.pct}%`,
                          height: 8,
                          background: "#2563eb",
                          borderRadius: 999,
                          transition: "width 120ms linear",
                        }}
                      />
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        color: "#e2e8f0",
                        fontSize: 12,
                        marginTop: 6,
                      }}
                    >
                      <span>
                        {new Date(progress.current * 1000)
                          .toISOString()
                          .substring(11, 19)}
                      </span>
                      <span>
                        {new Date(progress.duration * 1000)
                          .toISOString()
                          .substring(11, 19)}
                      </span>
                    </div>
                  </div>
                )}
                {/* Overlay with live recognition during preview */}
                {latestRecognition && latestRecognition.name && (
                  <div
                    style={{
                      position: "absolute",
                      top: 12,
                      right: 12,
                      background: "#e0e7ff",
                      border: "2px solid #2563eb",
                      borderRadius: 10,
                      padding: "10px 12px",
                      color: "#1e293b",
                      boxShadow: "0 2px 8px rgba(37,99,235,0.15)",
                      minWidth: 180,
                      textAlign: "center",
                    }}
                  >
                    {(() => {
                      const nameStr = latestRecognition.name || "";
                      const match = nameStr.match(/(\d{6,})/);
                      const studentNumber = match ? match[1] : "-";
                      const fullName = match
                        ? nameStr
                            .replace(studentNumber, "")
                            .replace(/[_-]+$/g, "")
                            .replace(/[_-]+/g, " ")
                            .trim()
                        : nameStr.replace(/[_-]+/g, " ").trim();
                      return (
                        <>
                          <div
                            style={{
                              fontWeight: 700,
                              fontSize: 15,
                              direction: "ltr",
                            }}
                          >
                            {fullName}
                          </div>
                          <div
                            style={{
                              color: "#2563eb",
                              fontWeight: 700,
                              direction: "ltr",
                            }}
                          >
                            {studentNumber}
                          </div>
                          <div style={{ fontSize: 12, color: "#334155" }}>
                            {latestRecognition.timestamp}
                          </div>
                        </>
                      );
                    })()}
                  </div>
                )}
                <div
                  style={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    width: "100%",
                    padding: "12px 0 8px 0",
                    background:
                      "linear-gradient(0deg, #181a20 80%, transparent 100%)",
                    textAlign: "center",
                    fontSize: 15,
                    color: "#fff",
                    fontWeight: 600,
                    letterSpacing: "0.5px",
                    borderBottomLeftRadius: 18,
                    borderBottomRightRadius: 18,
                  }}
                >
                  تتم المعالجة بالتزامن مع المعاينة
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button
              className="btn"
              onClick={() => handleUpload()}
              disabled={!file || loading}
            >
              إعادة الرفع/المعالجة
            </button>
            <button
              className="btn"
              onClick={() => {
                setFile(null);
                setRecognized([]);
                setAbsent([]);
                setLatestRecognition(null);
                setIsProcessing(false);
                setProgress({ current: 0, duration: 0, pct: 0 });
              }}
            >
              إلغاء المعاينة
            </button>
            {isProcessing && (
              <span
                style={{
                  padding: "8px 10px",
                  background: "#2563eb",
                  color: "#fff",
                  borderRadius: 6,
                  boxShadow: "0 2px 8px rgba(37,99,235,0.15)",
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                المعالجة قيد التنفيذ
              </span>
            )}
          </div>

          {videoInfo && (
            <p style={{ marginTop: 10, color: "#2563eb", fontWeight: 500 }}>
              تم الرفع: {videoInfo.path}
            </p>
          )}
        </div>

        {Boolean(recognized.length || absent.length) && (
          <div
            style={{
              display: "flex",
              gap: "18px",
              marginTop: 18,
              justifyContent: "center",
            }}
          >
            <AttendanceTable recognized={recognized} />
            <AbsenceTable absent={absent} />
          </div>
        )}
      </div>

      {/* Sidebar - Latest Recognition */}
      <LatestRecognitionCard
        latestRecognition={latestRecognition}
        isMonitoring={isProcessing}
      />
    </div>
  );
}
