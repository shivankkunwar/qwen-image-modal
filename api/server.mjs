// Tiny local proxy in front of the deployed Modal API.
// Adds your Modal-Key / Modal-Secret to every request so they never reach
// browser code. Forwards /v1/*, /health and the interactive docs unchanged,
// so new API features work here without touching this file.
// Zero dependencies (Node 20.6+ for --env-file).
//
//   node --env-file=api/.env api/server.mjs
//   open http://127.0.0.1:8787/docs

import { createServer } from "node:http";
import { Readable } from "node:stream";

const { MODAL_API_URL, MODAL_KEY, MODAL_SECRET, CORS_ORIGIN } = process.env;
const PORT = Number(process.env.PORT ?? 8787);
const HOST = process.env.HOST ?? "127.0.0.1"; // localhost only: anyone who can reach this spends your credits

if (!MODAL_API_URL || !MODAL_KEY || !MODAL_SECRET) {
  console.error("Missing MODAL_API_URL / MODAL_KEY / MODAL_SECRET (see api/.env.example)");
  process.exit(1);
}

const FORWARDED_PATHS = /^\/(v1\/|health$|docs$|openapi\.json$)/;
const MAX_BODY_BYTES = 250 * 1024 * 1024; // 10 reference images as base64, with room to spare
const PASS_HEADERS = ["content-type", "cache-control", "content-disposition"];

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error("request body too large"), { status: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function corsHeaders(req) {
  // Only for a local dev UI on another port, e.g. CORS_ORIGIN=http://localhost:3000
  if (!CORS_ORIGIN || req.headers.origin !== CORS_ORIGIN) return {};
  return {
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

async function proxy(req, res) {
  const path = new URL(req.url, "http://x").pathname;
  if (!FORWARDED_PATHS.test(path)) {
    res.writeHead(404, { "Content-Type": "application/json", ...corsHeaders(req) });
    return res.end(JSON.stringify({ error: "not found" }));
  }

  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  const t0 = Date.now();
  // fetch follows Modal's 303 "still working" redirects (sent after 150 s, e.g.
  // during a cold start) and keeps our headers because they're same-origin.
  const upstream = await fetch(new URL(req.url, MODAL_API_URL), {
    method: req.method,
    headers: {
      "Content-Type": req.headers["content-type"] ?? "application/json",
      "Modal-Key": MODAL_KEY,
      "Modal-Secret": MODAL_SECRET,
    },
    body: hasBody ? await readBody(req) : undefined,
    signal: AbortSignal.timeout(20 * 60 * 1000), // matches the GPU function timeout
  });

  const headers = { ...corsHeaders(req) };
  for (const name of PASS_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) headers[name] = value;
  }
  res.writeHead(upstream.status, headers);
  console.log(`${req.method} ${req.url} -> ${upstream.status} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  if (upstream.body) Readable.fromWeb(upstream.body).pipe(res);
  else res.end();
}

createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, corsHeaders(req));
    return res.end();
  }
  try {
    await proxy(req, res);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) {
      res.writeHead(err.status ?? 502, { "Content-Type": "application/json", ...corsHeaders(req) });
      res.end(JSON.stringify({ error: String(err.message ?? err) }));
    } else {
      res.end();
    }
  }
}).listen(PORT, HOST, () => {
  console.log(`Qwen image API proxy on http://${HOST}:${PORT}  (docs: /docs)`);
});
