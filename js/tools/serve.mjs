// Static server for the demo, on 127.0.0.1 only. Serves js/ and maps
// /models/student.int8.onnx to the ONNX file given by $OPENPRONOUNCE_ONNX. Sends the
// cross-origin isolation headers multi-threaded WASM needs, and gzip when the client asks
// (except for the ONNX model, sent as is, like a CDN that does not compress binaries).
//   OPENPRONOUNCE_ONNX=/mnt/t7/openpronounce-browser/onnx/student.int8.onnx node tools/serve.mjs [port]
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createGzip } from "node:zlib";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.argv[2] ?? 8765);
const onnx = process.env.OPENPRONOUNCE_ONNX ?? "/mnt/t7/openpronounce-browser/onnx/student.int8.onnx";
const clips = process.env.OPENPRONOUNCE_CLIPS; // optional directory of test clips, served under /clips/
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".gz": "application/gzip", ".wav": "audio/wav",
  ".css": "text/css", ".svg": "image/svg+xml" };

createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  let file;
  if (path === "/models/student.int8.onnx") file = onnx;
  else if (clips && path.startsWith("/clips/")) file = join(clips, normalize(path.slice(7)).replace(/^(\.\.(\/|\\|$))+/, ""));
  else file = join(root, normalize(path).replace(/^(\.\.(\/|\\|$))+/, ""));
  if (path === "/") {
    res.writeHead(302, { Location: "/demo/" });
    return res.end();
  }
  if (!file.startsWith(root + sep) && file !== onnx && !(clips && file.startsWith(clips))) {
    res.writeHead(403);
    return res.end();
  }
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (!existsSync(file)) {
    res.writeHead(404);
    return res.end("not found");
  }
  const type = types[extname(file)] ?? "application/octet-stream";
  const headers = { "Content-Type": type, "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp", "Cache-Control": "no-cache" };
  const compress = /\bgzip\b/.test(req.headers["accept-encoding"] ?? "") && ![".onnx", ".gz", ".wav"].includes(extname(file));
  if (compress) {
    res.writeHead(200, { ...headers, "Content-Encoding": "gzip" });
    createReadStream(file).pipe(createGzip({ level: 6 })).pipe(res);
  } else {
    res.writeHead(200, { ...headers, "Content-Length": statSync(file).size });
    createReadStream(file).pipe(res);
  }
}).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/demo/`));
