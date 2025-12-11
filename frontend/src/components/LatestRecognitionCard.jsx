import React from "react";

export default function LatestRecognitionCard({
  latestRecognition,
  isMonitoring,
}) {
  return (
    <div
      className="card"
      style={{
        width: "320px",
        height: "fit-content",
        background: "#f8fafc",
        border: "1.5px solid #2563eb",
        boxShadow: "0 4px 16px rgba(37,99,235,0.08)",
        borderRadius: "16px",
        padding: "20px 18px 18px 18px",
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
              background: "#e0e7ff",
              borderRadius: "10px",
              marginBottom: "14px",
              boxShadow: "0 2px 8px rgba(37,99,235,0.07)",
              textAlign: "center",
              padding: "18px 12px 14px 12px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            {latestRecognition.image_url ? (
              <img
                src={`${latestRecognition.image_url}?t=${Date.now()}`}
                alt="آخر وجه تم التعرف عليه"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                  const fallback = document.getElementById(
                    "latest-face-fallback"
                  );
                  if (fallback) fallback.style.display = "flex";
                }}
                style={{
                  width: "120px",
                  height: "120px",
                  objectFit: "cover",
                  borderRadius: "10px",
                  marginBottom: "12px",
                  border: "2.5px solid #2563eb",
                  background: "#fff",
                }}
              />
            ) : null}
            <div
              id="latest-face-fallback"
              style={{
                display: latestRecognition.image_url ? "none" : "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "120px",
                height: "120px",
                borderRadius: "10px",
                marginBottom: "12px",
                border: "2.5px solid #2563eb",
                background: "#f8fafc",
                color: "#94a3b8",
                fontSize: 14,
              }}
              title="لا توجد صورة متاحة"
            >
              لا توجد صورة
            </div>
            {/* استخراج الاسم ورقم الطالب */}
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
                      fontSize: "20px",
                      fontWeight: "bold",
                      color: "#1e293b",
                      marginBottom: "4px",
                      wordBreak: "break-word",
                      letterSpacing: "0.5px",
                      direction: "ltr",
                    }}
                  >
                    {fullName}
                  </div>
                  <div
                    style={{
                      fontSize: "18px",
                      fontWeight: 600,
                      color: "#2563eb",
                      marginBottom: "7px",
                      letterSpacing: "1px",
                      direction: "ltr",
                    }}
                  >
                    {studentNumber}
                  </div>
                </>
              );
            })()}
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
          {isMonitoring ? "في انتظار التعرف على وجه..." : "لا توجد بيانات بعد"}
        </div>
      )}
    </div>
  );
}
