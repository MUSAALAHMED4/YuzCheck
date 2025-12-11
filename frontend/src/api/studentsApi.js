import client from "./client";

export const listStudents = async () => {
  const { data } = await client.get("/api/students");
  return data;
};

export const importStudents = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post("/api/students/import", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};
