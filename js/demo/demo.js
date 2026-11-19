// Demo page: loads the ONNX student and the scoring data from this server, records or
// reads a clip, and shows the score and the words to work on.
//   ?ep=webgpu       try WebGPU first (falls back to WASM for unsupported nodes)
//   ?model=<url>     another ONNX file (default /models/student.int8.onnx)
//   ?clip=<url>&text=<sentence>   score a clip from the server right away (for testing)
import { createAssessor, decodeToMono16k } from "../src/index.js";

const params = new URLSearchParams(location.search);
const useWebGpu = params.get("ep") === "webgpu";
const modelUrl = params.get("model") ?? "/models/student.int8.onnx";
const dist = "/node_modules/onnxruntime-web/dist/";

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };
const progress = (fraction) => { $("bar").firstElementChild.style.width = `${Math.round(fraction * 100)}%`; };

const EXAMPLES = [
  "The quick brown fox jumps over the lazy dog",
  "I would like a cup of coffee, please",
  "She sells seashells by the seashore",
  "We are going to the theatre on Thursday",
];
for (const text of EXAMPLES) {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.textContent = text;
  chip.onclick = () => { $("sentence").value = text; };
  $("examples").append(chip);
}

const metrics = { loadStart: performance.now() };
window.__openpronounce = metrics;

async function fetchWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      progress(received / total);
      status(`Downloading the model: ${(received / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`);
    }
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

let assessor = null;

async function load() {
  const ort = await import(dist + (useWebGpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
  ort.env.wasm.wasmPaths = dist;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const modelBytes = await fetchWithProgress(modelUrl);
  metrics.downloadMs = performance.now() - metrics.loadStart;
  status("Preparing the model…");
  const providers = useWebGpu && navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  assessor = await createAssessor({ ort, modelBytes, dataUrl: "/models/", executionProviders: providers });
  metrics.readyMs = performance.now() - metrics.loadStart;
  metrics.assessor = assessor;
  metrics.providers = providers;
  metrics.threads = ort.env.wasm.numThreads;
  progress(1);
  $("bar").hidden = true;
  status(`Ready. Type or pick a sentence, then record yourself reading it.`);
  $("record").disabled = false;
  $("file").disabled = false;
  if (params.get("clip")) {
    if (params.get("text")) $("sentence").value = params.get("text");
    const buffer = await (await fetch(params.get("clip"))).arrayBuffer();
    await score(buffer);
  }
}

async function score(arrayBuffer) {
  const sentence = $("sentence").value.trim();
  if (!sentence) {
    status("Type the sentence you are going to read first.");
    return;
  }
  status("Listening to your recording…");
  const t0 = performance.now();
  const samples = await decodeToMono16k(arrayBuffer);
  const t1 = performance.now();
  const result = await assessor.assess(samples, sentence);
  metrics.last = { seconds: samples.length / 16000, decodeMs: t1 - t0, ...result.timings, totalMs: performance.now() - t0 };
  metrics.result = result;
  render(result, sentence);
  status("Done. Record again whenever you like.");
}

const escape = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function render(result, sentence) {
  $("result").hidden = false;
  const scores = result.scores ?? {};
  $("total").textContent = result.score === null ? "?" : Math.round(result.score);
  for (const key of ["accuracy", "fluency", "prosodic"]) $(key).textContent = scores[key] === undefined ? "?" : Math.round(scores[key]);
  const reasons = result.reasons ?? [];
  $("reasons").hidden = !reasons.length;
  $("reasons").textContent = reasons.length ? `Take this score with care: ${reasons.join("; ")}.` : "";

  const words = result.differences.words ?? [];
  const byPosition = new Map(words.map((w) => [w.position, w]));
  // Show the sentence as typed, marking the words the model flags.
  const tokens = sentence.split(/(\s+)/);
  let position = 0;
  const html = tokens.map((token) => {
    if (/^\s+$/.test(token)) return token;
    if (!/[\p{L}\p{N}]/u.test(token)) return escape(token);
    const report = byPosition.get(position++);
    const bad = report?.mispronounced;
    const title = report ? `expected /${report.expected.join("")}/, heard /${report.actual.join("")}/` : "";
    return `<span class="w${bad ? " bad" : ""}" title="${escape(title)}">${escape(token)}</span>`;
  }).join("");
  $("words").innerHTML = html;

  const flagged = words.filter((w) => w.mispronounced);
  $("details").innerHTML = flagged.length
    ? flagged.map((w) => {
      const phones = w.phones.filter((p) => p.confidence >= 0.5)
        .map((p) => (p.heard ? `/${escape(p.expected)}/ sounded like /${escape(p.heard)}/` : `/${escape(p.expected)}/ was not heard`));
      const what = w.missing ? "not said" : phones.join(", ") || "sounds unclear";
      return `<div><b>${escape(w.word)}</b> <span class="ipa">/${escape(w.expected.join(""))}/</span>: ${what}</div>`;
    }).join("")
    : "<div>No word stands out. Nicely done.</div>";
  const t = metrics.last;
  $("timing").textContent = `${t.seconds.toFixed(1)} s of audio, model ${Math.round(t.model_ms)} ms, scoring ${Math.round(t.scoring_ms)} ms`
    + ` (${metrics.providers.join(" + ")}, ${metrics.threads} thread${metrics.threads > 1 ? "s" : ""}).`;
}

$("file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) await score(await file.arrayBuffer()).catch(fail);
});

let recorder = null;
$("record").addEventListener("click", async () => {
  if (recorder) {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      recorder = null;
      $("record").textContent = "Record";
      $("record").classList.remove("recording");
      await score(await new Blob(chunks).arrayBuffer()).catch(fail);
    };
    recorder.start();
    $("record").textContent = "Stop";
    $("record").classList.add("recording");
    status("Recording. Read the sentence, then press Stop.");
  } catch (error) {
    fail(error);
  }
});

function fail(error) {
  console.error(error);
  status(`Something went wrong: ${error.message ?? error}`);
}

load().catch(fail);
