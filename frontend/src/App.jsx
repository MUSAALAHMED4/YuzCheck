import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage.jsx";
import AttendancePage from "./pages/AttendancePage.jsx";
import StudentsPage from "./pages/StudentsPage.jsx";
import FacesPage from "./pages/FacesPage.jsx";
import VideoUploadPage from "./pages/VideoUploadPage.jsx";

export default function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <h1>YuzCheck</h1>
        <div className="nav-links">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/attendance">Attendance</NavLink>
          <NavLink to="/students">Students</NavLink>
          <NavLink to="/faces">Faces</NavLink>
          <NavLink to="/upload-video">رفع فيديو</NavLink>
        </div>
      </nav>
      <div className="container">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/attendance" element={<AttendancePage />} />
          <Route path="/students" element={<StudentsPage />} />
          <Route path="/faces" element={<FacesPage />} />
          <Route path="/upload-video" element={<VideoUploadPage />} />
        </Routes>
      </div>
    </div>
  );
}
