import React, { useState, useEffect } from "react";
import LatestRecognitionCard from "../components/LatestRecognitionCard";
import {
  uploadVideo,
  runAttendanceLive,
  getLatestRecognition,
} from "../api/attendanceApi";
import AttendanceTable from "../components/AttendanceTable";
import AbsenceTable from "../components/AbsenceTable";

export default function AttendancePage() {
  const [file, setFile] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [latestRecognition, setLatestRecognition] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const videoRef = React.useRef(null);
  const streamRef = React.useRef(null);

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

  // Setup local webcam preview during processing
  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Failed to open webcam preview:", err);
      }
    };
    if (loading) {
      setupCamera();
    }
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        try {
          videoRef.current.pause();
        } catch {}
        videoRef.current.srcObject = null;
      }
    };
  }, [loading]);

  return (
    <div style={{ display: "flex", gap: "16px" }}>
      {/* Main Content */}
      <div style={{ flex: 1 }}>
        <div className="card">
          <h2>أخذ الحضور من الكاميرا</h2>
          <button className="btn" onClick={handleRun} disabled={loading}>
            بدء
          </button>
          {loading && (
            <div style={{ marginTop: 12 }}>
              {/* Webcam Preview - full width */}
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
                  muted
                  playsInline
                  style={{
                    width: "98%",
                    height: "96%",
                    objectFit: "cover",
                    background: "#0b1220",
                    borderRadius: "14px",
                    boxShadow: "0 2px 12px rgba(0,0,0,0.10)",
                  }}
                />
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
                    textShadow: "0 2px 8px rgba(37,99,235,0.15)",
                    borderBottomLeftRadius: 18,
                    borderBottomRightRadius: 18,
                  }}
                >
                  ضع وجهك داخل الإطار ليتم التعرف عليه
                </div>
              </div>
            </div>
          )}
        </div>
        {result && (
          <div
            style={{
              display: "flex",
              gap: "18px",
              marginTop: 18,
              justifyContent: "center",
            }}
          >
            {/* جدول الحضور */}
            <AttendanceTable recognized={result.recognized} />
            {/* جدول الغياب */}
            <AbsenceTable absent={result.absent} />
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
      <LatestRecognitionCard
        latestRecognition={latestRecognition}
        isMonitoring={isMonitoring}
      />
    </div>
  );
}
