// runs before npm run deploy. the site on github pages is public, and it is built
// from whatever the pipeline left in src/data, so a head only ships when it is on
// public-subjects.json. anything else stops the deploy before it builds.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dataDir = join(root, "src", "data");
const allowed = JSON.parse(readFileSync(join(root, "public-subjects.json"), "utf8")).subjects;

const subjects = existsSync(dataDir)
  ? readdirSync(dataDir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && existsSync(join(dataDir, d.name, "markers.json")))
      .map((d) => d.name)
  : [];

if (!subjects.length) {
  console.error("no subjects in src/data, run python run_pipeline.py first");
  process.exit(1);
}

const blocked = subjects.filter((s) => !(s in allowed));
if (blocked.length) {
  console.error(`not deploying. these subjects are not on public-subjects.json: ${blocked.join(", ")}`);
  console.error("remove their generated folders from viewer/src/data, then regenerate the registry with run_pipeline.py --template before deploying");
  process.exit(1);
}

console.log(`public subjects ok: ${subjects.join(", ")}`);
// an aim 2 e field lives inside its subject's folder, so the list above already
// covers it. say which heads carry one, and whether it is the demo or a real run
const efield = subjects.filter((s) => existsSync(join(dataDir, s, "efield", "manifest.json")));
for (const s of efield) {
  const m = JSON.parse(readFileSync(join(dataDir, s, "efield", "manifest.json"), "utf8"));
  console.log(`e-field ships for ${s}${m.demo ? " (demo)" : ""}`);
}
if (existsSync(join(dataDir, "aim1_stats.json"))) {
  console.log("note: src/data/aim1_stats.json is there, so the Brainsight results (aggregate numbers,");
  console.log("no subject ids) go on the public site too. delete it before deploying if they should not");
}
