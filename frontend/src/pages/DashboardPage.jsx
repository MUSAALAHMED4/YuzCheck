import React, { useEffect, useState } from "react";
import { getSummary } from "../api/attendanceApi";

export default function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        setSummary(await getSummary());
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <div className="card">Loading summary…</div>;

  return (
    <div>
      <div className="card">
        <h2>Summary</h2>
        <p>Total students: {summary?.total ?? 0}</p>
        <p>Present: {summary?.present?.length ?? 0}</p>
        <p>Absent: {summary?.absent?.length ?? 0}</p>
      </div>
      <div className="row">
        <div className="card" style={{ flex: 1 }}>
          <h3>Present</h3>
          <ul>
            {(summary?.present || []).map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
        <div className="card" style={{ flex: 1 }}>
          <h3>Absent</h3>
          <ul>
            {(summary?.absent || []).map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
