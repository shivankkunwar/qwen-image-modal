// Tiny local API in front of the deployed Modal endpoint.
// Keeps MODAL_KEY / MODAL_SECRET on the server so they never reach browser code.
// Zero dependencies (Node 20.6+ for --env-file).
//
//   node --env-file=api/.env api/server.mjs
//
//   GET  /health
//   POST /generate   {"prompt": "...", "width"?, "height"?, "steps"?, "seed"?, "transparent"?}
//                    -> image/png

import { createServer } from "node:http";

const { MODAL_ENDPOINT_URL, MODAL_KEY, MODAL_SECRET } = process.env;
const PORT = Number(process.env.PORT ?? 8787);
const HOST = process.env.HOST ?? "127.0.0.1"; // localhost only: anyone who can reach this spends your credits

if (!MODAL_ENDPOINT_URL || !MODAL_KEY || !MODAL_SECRET) {
  console.error("Missing MODAL_ENDPOINT_URL / MODAL_KEY / MODAL_SECRET (see api/.env.example)");
  process.exit(1);
}

const ALLOWED = ["prompt", "width", "height", "steps", "seed", "transparent"];

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readJson(req) {
  let raw = "";
  for await (const chunk of req) {
    raw += chunk;
    if (raw.length > 100_000) throw new Error("body too large");
  }
  return JSON.parse(raw || "{}");
}

async function generate(req, res) {
  let body;
  try {
    body = await readJson(req);
  } catch {
    return sendJson(res, 400, { error: "body must be valid JSON" });
  }
  if (typeof body.prompt !== "string" || !body.prompt.trim()) {
    return sendJson(res, 400, { error: "prompt (non-empty string) is required" });
  }
  // Forward only known fields so callers can't smuggle anything else through.
  const payload = Object.fromEntries(ALLOWED.filter((k) => k in body).map((k) => [k, body[k]]));

  const t0 = Date.now();
  // fetch follows Modal's 303 "still working" redirects (sent after 150 s, e.g. on
  // a cold start) and keeps our headers because the redirect is same-origin.
  const upstream = await fetch(MODAL_ENDPOINT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Modal-Key": MODAL_KEY,
      "Modal-Secret": MODAL_SECRET,
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(15 * 60 * 1000), // matches the Modal function timeout
  });

  if (!upstream.ok) {
    const detail = await upstream.text();
    console.error(`Modal ${upstream.status}: ${detail}`);
    return sendJson(res, upstream.status, { error: "generation failed", status: upstream.status, detail });
  }

  const png = Buffer.from(await upstream.arrayBuffer());
  console.log(`generated ${png.length} bytes in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  res.writeHead(200, { "Content-Type": "image/png", "Content-Length": png.length });
  res.end(png);
}

createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/health") return sendJson(res, 200, { ok: true });
    if (req.method === "POST" && req.url === "/generate") return await generate(req, res);
    sendJson(res, 404, { error: "not found" });
  } catch (err) {
    console.error(err);
    if (!res.headersSent) sendJson(res, 502, { error: String(err.message ?? err) });
  }
}).listen(PORT, HOST, () => {
  console.log(`Qwen image API on http://${HOST}:${PORT}  (POST /generate)`);
});
