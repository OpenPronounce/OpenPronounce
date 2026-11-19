// Write models/cmudict.txt.gz: the first pronunciation of every CMUdict entry, comments
// dropped, from the copy shipped with the Python package (BSD-style licence, see
// openpronounce/data/cmudict.LICENSE, copied next to it).
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync, constants } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const source = process.argv[2] ?? join(here, "..", "..", "openpronounce", "data", "cmudict.dict.gz");
const out = join(here, "..", "models");
mkdirSync(out, { recursive: true });
const lines = [];
for (let line of gunzipSync(readFileSync(source)).toString("utf8").split("\n")) {
  line = line.split("#")[0].trim();
  if (!line) continue;
  const [word, ...symbols] = line.split(/\s+/);
  if (!word.includes("(")) lines.push(`${word} ${symbols.join(" ")}`);
}
const gz = gzipSync(Buffer.from(lines.join("\n") + "\n"), { level: constants.Z_BEST_COMPRESSION });
writeFileSync(join(out, "cmudict.txt.gz"), gz);
copyFileSync(join(dirname(source), "cmudict.LICENSE"), join(out, "cmudict.LICENSE"));
console.log(`${lines.length} entries, ${(gz.length / 1024).toFixed(0)} KB gzip -> models/cmudict.txt.gz`);
