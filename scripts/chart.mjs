/**
 * Draw learning curves from one or more training runs into a single PNG.
 *
 *   node scripts/chart.mjs runs/champion                    # one run directory
 *   node scripts/chart.mjs runs/* --out runs/curves.png     # several, overlaid
 *
 * Each run directory must contain run.json (written by scripts/train.mjs) or at
 * least the per-lane history files.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Bitmap } from './render.mjs';

const FONT = {
  '0': ['111', '101', '101', '101', '111'],
  '1': ['010', '110', '010', '010', '111'],
  '2': ['111', '001', '111', '100', '111'],
  '3': ['111', '001', '111', '001', '111'],
  '4': ['101', '101', '111', '001', '001'],
  '5': ['111', '100', '111', '001', '111'],
  '6': ['111', '100', '111', '101', '111'],
  '7': ['111', '001', '010', '010', '010'],
  '8': ['111', '101', '111', '101', '111'],
  '9': ['111', '101', '111', '001', '111'],
  'a': ['010', '101', '111', '101', '101'],
  b: ['110', '101', '110', '101', '110'],
  c: ['011', '100', '100', '100', '011'],
  d: ['110', '101', '101', '101', '110'],
  e: ['111', '100', '110', '100', '111'],
  f: ['111', '100', '110', '100', '100'],
  g: ['011', '100', '101', '101', '011'],
  h: ['101', '101', '111', '101', '101'],
  i: ['111', '010', '010', '010', '111'],
  j: ['001', '001', '001', '101', '010'],
  k: ['101', '110', '100', '110', '101'],
  l: ['100', '100', '100', '100', '111'],
  m: ['101', '111', '111', '101', '101'],
  n: ['101', '111', '111', '111', '101'],
  o: ['010', '101', '101', '101', '010'],
  p: ['110', '101', '110', '100', '100'],
  q: ['010', '101', '101', '111', '011'],
  r: ['110', '101', '110', '101', '101'],
  s: ['011', '100', '010', '001', '110'],
  t: ['111', '010', '010', '010', '010'],
  u: ['101', '101', '101', '101', '111'],
  v: ['101', '101', '101', '101', '010'],
  w: ['101', '101', '111', '111', '101'],
  x: ['101', '101', '010', '101', '101'],
  y: ['101', '101', '010', '010', '010'],
  z: ['111', '001', '010', '100', '111'],
  '.': ['000', '000', '000', '000', '010'],
  ',': ['000', '000', '000', '010', '100'],
  '-': ['000', '000', '111', '000', '000'],
  '/': ['001', '001', '010', '100', '100'],
  ':': ['000', '010', '000', '010', '000'],
  ' ': ['000', '000', '000', '000', '000'],
  '(': ['001', '010', '010', '010', '001'],
  ')': ['100', '010', '010', '010', '100'],
  '=': ['000', '111', '000', '111', '000'],
  '+': ['000', '010', '111', '010', '000'],
};

/** Draw a tiny 3x5 bitmap font (scaled), so the chart needs no font files. */
function drawText(target, text, x, y, color, scale = 1) {
  let cx = Math.round(x);
  for (const ch of String(text).toLowerCase()) {
    const glyph = FONT[ch] || FONT[' '];
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 3; col++) {
        if (glyph[row][col] !== '1') continue;
        for (let sy = 0; sy < scale; sy++) {
          for (let sx = 0; sx < scale; sx++) {
            target.blend(cx + col * scale + sx, Math.round(y) + row * scale + sy, color, 1);
          }
        }
      }
    }
    cx += 4 * scale;
  }
}

function formatSteps(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(Math.round(n));
}
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const argv = process.argv.slice(2);
const dirs = [];
let out = 'runs/learning-curves.png';
let title = 'RL--RUNNING - learning curves';
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--out') out = argv[++i];
  else if (argv[i] === '--title') title = argv[++i];
  else dirs.push(argv[i]);
}
if (!dirs.length) dirs.push('runs/champion');

const LANE_COLORS = [
  [76, 155, 255],
  [67, 209, 124],
  [240, 180, 41],
  [199, 125, 255],
];

/** Load per-lane curves from a run directory. */
function loadRun(dir) {
  const full = resolve(root, dir);
  if (!existsSync(full)) return null;
  const runPath = join(full, 'run.json');
  if (existsSync(runPath)) {
    const run = JSON.parse(readFileSync(runPath, 'utf8'));
    return { dir, lanes: run.lanes || [] };
  }
  // Fall back to lane*-history.json files.
  const lanes = [];
  for (const file of readdirSync(full)) {
    const m = file.match(/^lane(\d+)-history\.json$/);
    if (!m) continue;
    lanes.push({ lane: Number(m[1]), curve: JSON.parse(readFileSync(join(full, file), 'utf8')) });
  }
  return lanes.length ? { dir, lanes } : null;
}

const runs = dirs.map(loadRun).filter(Boolean);
if (!runs.length) {
  console.error(`no runs found in: ${dirs.join(', ')}`);
  process.exit(1);
}

const W = 1360;
const H = 560;
const padL = 74;
const padR = 26;
const padT = 76;
const padB = 64;
const bmp = new Bitmap(W, H, [13, 16, 24]);

const maxStep = Math.max(1000, ...runs.flatMap((r) => r.lanes.flatMap((l) => (l.curve || []).map((p) => p[0] || 0))));
const maxDist = Math.max(1, ...runs.flatMap((r) => r.lanes.flatMap((l) => (l.curve || []).map((p) => p[1] || 0))));
const X = (step) => padL + (step / maxStep) * (W - padL - padR);
const Y = (dist) => H - padB - (dist / (maxDist * 1.1)) * (H - padT - padB);

// grid + labels -----------------------------------------------------------
for (let i = 0; i <= 5; i++) {
  const y = Y((maxDist * 1.1 * i) / 5);
  bmp.fillRect(padL, y, W - padR, y, [30, 40, 60]);
  drawText(bmp, `${((maxDist * 1.1 * i) / 5).toFixed(0)} m`, 8, y - 6, [150, 170, 205]);
}
for (let i = 0; i <= 5; i++) {
  const x = X((maxStep * i) / 5);
  bmp.fillRect(x, padT, x, H - padB, [26, 34, 52]);
  drawText(bmp, formatSteps((maxStep * i) / 5), x - 14, H - padB + 18, [150, 170, 205]);
}
drawText(bmp, title.toUpperCase(), padL, 26, [220, 232, 252], 2);
drawText(bmp, 'training steps', W / 2 - 28, H - 14, [140, 158, 190]);

// curves -----------------------------------------------------------------
runs.forEach((run, runIndex) => {
  run.lanes.forEach((lane, laneIndex) => {
    const curve = lane.curve || [];
    if (curve.length < 2) return;
    const color = LANE_COLORS[(lane.lane ?? laneIndex) % LANE_COLORS.length];
    // faint per-episode line
    for (let i = 1; i < curve.length; i++) {
      bmp.line(X(curve[i - 1][0]), Y(curve[i - 1][1]), X(curve[i][0]), Y(curve[i][1]), color, 0.22);
    }
    // running best
    let best = 0;
    let prev = null;
    for (const [step, dist] of curve) {
      best = Math.max(best, dist);
      const point = [X(step), Y(best)];
      if (prev) for (let t = 0; t < 2; t++) bmp.line(prev[0], prev[1] + t, point[0], point[1] + t, color, 1);
      prev = point;
    }
    // Legend: one entry per lane, laid out in a row under the title.
    const perRow = Math.ceil(runs.reduce((n, r) => n + r.lanes.length, 0) / 2);
    const slot = runIndex * 0 + (laneIndex + runs.slice(0, runIndex).reduce((n, r) => n + r.lanes.length, 0));
    const col = slot % Math.max(1, perRow);
    const row = Math.floor(slot / Math.max(1, perRow));
    const lx = padL + col * ((W - padL - padR) / Math.max(1, perRow));
    const ly = padT - 40 + row * 16;
    bmp.fillRect(lx, ly, lx + 14, ly + 9, color);
    const name = `${run.dir.split('/').pop()} lane ${lane.lane ?? laneIndex}`;
    drawText(bmp, `${name} best ${(lane.best ?? best).toFixed(1)} m`, lx + 20, ly + 8, [206, 220, 244]);
  });
});

const outPath = resolve(root, out);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, bmp.toPNG());
console.log(`wrote ${out} (${runs.length} run(s), ${maxStep} steps wide)`);

