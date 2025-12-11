import client from "./client";

export const uploadVideo = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post("/api/video/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const runAttendance = async (payload) => {
  const { data } = await client.post("/api/attendance/run", payload);
  return data;
};

export const runAttendanceLive = async (payload = {}) => {
  const { data } = await client.post("/api/attendance/live", payload);
  return data;
};

export const getSummary = async () => {
  const { data } = await client.get("/api/attendance/summary");
  return data;
};

export const getLatestRecognition = async () => {
  const { data } = await client.get("/api/attendance/latest");
  return data;
};
