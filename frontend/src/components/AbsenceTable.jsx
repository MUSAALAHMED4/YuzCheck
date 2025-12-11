import React from "react";

export default function AbsenceTable({ absent = [] }) {
  return (
    <div
      className="card"
      style={{
        width: 480,
        minWidth: 320,
        background: "#181a20",
        boxShadow: "0 2px 8px #0001",
      }}
    >
      <h3 style={{ color: "#dc2626", marginBottom: 10, fontWeight: 700 }}>
        الغياب
      </h3>
      <div
        style={{
          maxHeight: 400,
          overflowY: "auto",
          borderRadius: 10,
          background: "#fff",
        }}
      >
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 15,
            background: "#fff",
          }}
        >
          <thead>
            <tr style={{ background: "#fff" }}>
              <th
                style={{
                  textAlign: "right",
                  padding: "10px 12px",
                  fontWeight: 700,
                  borderBottom: "2px solid #e5e7eb",
                  color: "#22292f",
                }}
              >
                الاسم الكامل{" "}
                <span
                  style={{ fontSize: 15, color: "#bdbdbd", marginRight: 4 }}
                ></span>
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: "10px 12px",
                  fontWeight: 700,
                  borderBottom: "2px solid #e5e7eb",
                  color: "#22292f",
                }}
              >
                رقم الطالب{" "}
                <span
                  style={{ fontSize: 15, color: "#bdbdbd", marginRight: 4 }}
                ></span>
              </th>
            </tr>
          </thead>
          <tbody>
            {absent.length === 0 ? (
              <tr>
                <td
                  colSpan={2}
                  style={{
                    textAlign: "center",
                    color: "#64748b",
                    padding: 14,
                    background: "#f3f4f6",
                  }}
                >
                  لا يوجد غياب
                </td>
              </tr>
            ) : (
              absent.map((n, idx) => {
                const nameStr = n || "";
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
                  <tr
                    key={idx}
                    style={{ background: idx % 2 === 0 ? "#f3f4f6" : "#fff" }}
                  >
                    <td
                      style={{
                        padding: "10px 12px",
                        direction: "ltr",
                        borderBottom: "1px solid #e5e7eb",
                        color: "#181a20",
                        fontWeight: 500,
                      }}
                    >
                      {fullName}
                    </td>
                    <td
                      style={{
                        padding: "10px 12px",
                        direction: "ltr",
                        borderBottom: "1px solid #e5e7eb",
                        color: "#2563eb",
                        fontWeight: 600,
                      }}
                    >
                      {studentNumber}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
