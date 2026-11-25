// Vercel serverless function used by the dashboard's AI features (Ask AI, LLM reasoner).
// GET  /api/ai -> {ok, model}            health check
// POST /api/ai {messages, tools} -> {message}  one OpenAI-style chat completion via Groq
//
// The key is read from the GROQ_API_KEY environment variable. Never commit it.
const MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-20b";
const URL = "https://api.groq.com/openai/v1/chat/completions";

module.exports = async function handler(req, res) {
  const key = process.env.GROQ_API_KEY;
  if (req.method === "GET") {
    return res.status(key ? 200 : 503).json({ ok: Boolean(key), model: MODEL });
  }
  if (req.method !== "POST") return res.status(405).json({ error: "method not allowed" });
  if (!key) return res.status(503).json({ error: "GROQ_API_KEY is not set" });

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  const payload = { model: MODEL, messages: body.messages || [], temperature: 0.2 };
  if (Array.isArray(body.tools) && body.tools.length) payload.tools = body.tools;

  try {
    const r = await fetch(URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (!r.ok) return res.status(r.status).json({ error: (j.error && j.error.message) || r.statusText });
    return res.status(200).json({ message: j.choices[0].message });
  } catch (e) {
    return res.status(502).json({ error: String((e && e.message) || e) });
  }
};
