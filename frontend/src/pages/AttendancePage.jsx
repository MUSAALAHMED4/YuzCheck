import React, { useState } from "react";
import { uploadVideo, runAttendanceLive } from "../api/attendanceApi";

export default function AttendancePage() {
  const [file, setFile] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);

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
    try {
      const res = await runAttendanceLive({ max_seconds: 30 });
      setResult(res);
      setShowSummary(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
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
  );
}
