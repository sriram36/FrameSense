// In Docker: nginx proxies /api → backend (VITE_API_URL is empty, prefix=/api).
// On Vercel: vercel.json rewrites /api → Render backend (prefix=/api).
export const API = import.meta.env.VITE_API_URL || "/api";

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API}/analyze`, { method: "POST", body: formData });
  return handle(res);
}

export async function getResult(id) {
  const res = await fetch(`${API}/results/${id}`);
  return handle(res);
}

export async function listResults(limit = 20, offset = 0) {
  const res = await fetch(`${API}/results?limit=${limit}&offset=${offset}`);
  return handle(res);
}

export async function getHealth() {
  const res = await fetch(`${API}/health`);
  return handle(res);
}
