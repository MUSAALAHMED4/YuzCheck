import React, { useEffect, useState } from "react";
import { listStudents, importStudents } from "../api/studentsApi";

export default function StudentsPage() {
  const [students, setStudents] = useState([]);
  const [file, setFile] = useState(null);
  const refresh = async () => {
    const data = await listStudents();
    setStudents(data.students);
  };
  useEffect(() => {
    refresh();
  }, []);

  const handleImport = async () => {
    if (!file) return;
    await importStudents(file);
    setFile(null);
    await refresh();
  };

  return (
    <div>
      <div className="card">
        <h2>Import Students</h2>
        <input
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <div style={{ marginTop: 8 }}>
          <button className="btn" onClick={handleImport} disabled={!file}>
            Upload
          </button>
        </div>
      </div>
      <div className="card">
        <h2>Students</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
            </tr>
          </thead>
          <tbody>
            {students.map((s, i) => (
              <tr key={i}>
                <td>{s.name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
