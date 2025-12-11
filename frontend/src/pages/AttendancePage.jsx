import React, { useState, useEffect } from "react";
import {
  uploadVideo,
  runAttendanceLive,
  getLatestRecognition,
} from "../api/attendanceApi";

export default function AttendancePage() {
  const [file, setFile] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [latestRecognition, setLatestRecognition] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);

  // Fetch latest recognition every 2 seconds when monitoring
  useEffect(() => {
    let interval;
    if (isMonitoring) {
      interval = setInterval(async () => {
        try {
          const latest = await getLatestRecognition();
          if (latest && latest.name) {
            setLatestRecognition(latest);
          }
        } catch (error) {
          console.error("Failed to fetch latest recognition:", error);
        }
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMonitoring]);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const info = await uploadVideo(file);
      setVideoInfo(info);
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    // Trigger live attendance from webcam
    setLoading(true);
    setIsMonitoring(true); // Start monitoring
    try {
      const res = await runAttendanceLive({ max_seconds: 30 });
      setResult(res);
      setShowSummary(true);
    } finally {
      setLoading(false);
      setIsMonitoring(false); // Stop monitoring
    }
  };

  return (
    <div style={{ display: "flex", gap: "16px" }}>
      {/* Main Content */}
      <div style={{ flex: 1 }}>
        <div className="card">
          <h2>اختيار فيديو (اختياري)</h2>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <div style={{ marginTop: 8 }}>
            <button
              className="btn"
              onClick={handleUpload}
              disabled={!file || loading}
            >
              رفع الفيديو
            </button>
          </div>
          {videoInfo && <p>تم الرفع: {videoInfo.path}</p>}
        </div>
        <div className="card">
          <h2>أخذ الحضور من الكاميرا</h2>
          <button className="btn" onClick={handleRun} disabled={loading}>
            بدء
          </button>
          {loading && <p>Processing…</p>}
        </div>
        {result && (
          <div className="row">
            <div className="card" style={{ flex: 1 }}>
              <h3>Recognized</h3>
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recognized.map((r, idx) => (
                    <tr key={idx}>
                      <td>{r.name}</td>
                      <td>{r.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card" style={{ flex: 1 }}>
              <h3>Absent</h3>
              <ul>
                {result.absent.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {result && showSummary && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.4)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
            }}
            onClick={() => setShowSummary(false)}
          >
            <div
              className="card"
              style={{ width: 520, maxHeight: "80vh", overflow: "auto" }}
              onClick={(e) => e.stopPropagation()}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <h3 style={{ margin: 0 }}>الطلاب الذين تم التعرف عليهم</h3>
                <button className="btn" onClick={() => setShowSummary(false)}>
                  إغلاق
                </button>
              </div>
              <p style={{ marginTop: 8 }}>الإجمالي: {result.present.length}</p>
              <table style={{ width: "100%" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>الاسم</th>
                    <th style={{ textAlign: "left" }}>الرقم</th>
                  </tr>
                </thead>
                <tbody>
                  {result.present.map((name) => {
                    const match = String(name).match(/(\d{6,})/);
                    const num = match ? match[1] : "-";
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{num}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Sidebar - Latest Recognition */}
      <div
        className="card"
        style={{
          width: "320px",
          position: "sticky",
          top: "24px",
          height: "fit-content",
          background: "#f8fafc",
          border: "1.5px solid #2563eb",
          boxShadow: "0 4px 16px rgba(37,99,235,0.08)",
          borderRadius: "16px",
          padding: "20px 18px 18px 18px",
          zIndex: 10,
        }}
      >
        <h3
          style={{
            marginTop: 0,
            marginBottom: 18,
            color: "#2563eb",
            fontWeight: 700,
            fontSize: "20px",
            letterSpacing: "0.5px",
          }}
        >
          آخر وجه تم التعرف عليه
        </h3>
        {isMonitoring && (
          <div
            style={{
              padding: "7px 0",
              background: "#2563eb",
              color: "#fff",
              borderRadius: "6px",
              marginBottom: "14px",
              textAlign: "center",
              fontSize: "15px",
              fontWeight: 500,
              boxShadow: "0 2px 8px rgba(37,99,235,0.10)",
              letterSpacing: "0.5px",
            }}
          >
            <span style={{ fontSize: "16px", marginRight: "6px" }}>🔴</span>{" "}
            مراقبة نشطة
          </div>
        )}
        {latestRecognition && latestRecognition.name ? (
          <div>
            <div
              style={{
                padding: "18px 12px 14px 12px",
                background: "#e0e7ff",
                borderRadius: "10px",
                marginBottom: "14px",
                boxShadow: "0 2px 8px rgba(37,99,235,0.07)",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: "22px",
                  fontWeight: "bold",
                  color: "#1e293b",
                  marginBottom: "7px",
                  wordBreak: "break-all",
                  letterSpacing: "0.5px",
                }}
              >
                {latestRecognition.name}
              </div>
              <div
                style={{ fontSize: "15px", color: "#334155", fontWeight: 500 }}
              >
                {latestRecognition.timestamp}
              </div>
            </div>
            <div
              style={{
                fontSize: "13px",
                color: "#2563eb",
                textAlign: "center",
                fontWeight: 500,
                letterSpacing: "0.3px",
              }}
            >
              يتم التحديث تلقائياً
            </div>
          </div>
        ) : (
          <div
            style={{
              padding: "36px 0",
              textAlign: "center",
              color: "#64748b",
              fontSize: "15px",
              fontWeight: 500,
            }}
          >
            {isMonitoring
              ? "في انتظار التعرف على وجه..."
              : "لا توجد بيانات بعد"}
          </div>
        )}
      </div>
    </div>
  );
}
