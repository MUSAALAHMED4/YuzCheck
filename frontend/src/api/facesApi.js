import client from "./client";

export const addPerson = async (name, files) => {
  const form = new FormData();
  form.append("name", name);
  for (const f of files) form.append("files", f);
  const { data } = await client.post("/api/faces/add-person", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const listFaces = async () => {
  const { data } = await client.get("/api/faces/list");
  return data;
};
