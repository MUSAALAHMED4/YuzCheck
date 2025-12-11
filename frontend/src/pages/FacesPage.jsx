import React, { useEffect, useState } from "react";
import client from "../api/client";
import { addPerson, listFaces } from "../api/facesApi";

export default function FacesPage() {
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [faces, setFaces] = useState({});

  const refresh = async () => {
    setFaces(await listFaces());
  };
  useEffect(() => {
    refresh();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name || files.length === 0) return;
    await addPerson(name, files);
    setName("");
    setFiles([]);
    await refresh();
  };

  return (
    <div>
      <div className="card">
        <h2>Add Person</h2>
        <form onSubmit={handleSubmit}>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files || []))}
            />
            <button className="btn" type="submit">
              Add
            </button>
          </div>
        </form>
      </div>
      <div className="card">
        <h2>Known Faces</h2>
        {Object.keys(faces).length === 0 ? (
          <p>No faces yet.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ border: "1px solid #444", padding: 8 }}>Photo</th>
                <th style={{ border: "1px solid #444", padding: 8 }}>Name</th>
                <th style={{ border: "1px solid #444", padding: 8 }}>
                  Student Number
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(faces).map(([person, imgs]) => {
                // Extract name and student number from person string
                // Assume format: NAME_STUDENTNUMBER or NAME_STUDENTNUMBER_EXTRA
                let displayName = person;
                let studentNumber = "";
                const match = person.match(/(.+)_([0-9]{11,})/);
                if (match) {
                  displayName = match[1].replace(/_/g, " ");
                  studentNumber = match[2];
                }
                // Only show the first image for each person
                const p = imgs[0];
                return (
                  <tr key={person}>
                    <td
                      style={{
                        border: "1px solid #444",
                        padding: 8,
                        textAlign: "center",
                      }}
                    >
                      <img
                        src={
                          p?.startsWith("/")
                            ? `${client.defaults.baseURL}${p}`
                            : p
                        }
                        alt={displayName}
                        style={{
                          width: 64,
                          height: 64,
                          objectFit: "cover",
                          borderRadius: 8,
                        }}
                      />
                    </td>
                    <td style={{ border: "1px solid #444", padding: 8 }}>
                      {displayName}
                    </td>
                    <td style={{ border: "1px solid #444", padding: 8 }}>
                      {studentNumber}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
