/**
 * Render a policy's episode as a PNG figure, and print measurable gait stats.
 *
 * This is the command line "eyes" of the project: the top panel is a Muybridge
 * style flipbook (poses at fixed horizontal pitch, so the gait cycle is readable)
 * and the bottom panel is a world space stroboscopic trace (poses at their true
 * positions, so the distance actually covered is readable).
 *
 *   node scripts/gait-strip.mjs runs/champion/lane0-best.json --out runs/gait.png
 *   node scripts/gait-strip.mjs --random --seconds 4        # untrained reference
 */
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RunningEnv, CONTROL_DT, ACTION_SIZE, JOINTS } from '../src/env.js';
import { Policy } from '../src/net.js';
import { Bitmap, drawSnapshot, COLORS } from './render.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const args = {
  out: 'runs/gait.png',
  seconds: 8,
  fps: 12,
  width: 1400,
  poles: 16,
  policy: null,
  random: false,
  seed: 5,
  greedy: true,
  panel: 'both',
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--out') args.out = argv[++i];
  else if (a === '--seconds') args.seconds = Number(argv[++i]);
  else if (a === '--fps') args.fps = Number(argv[++i]);
  else if (a === '--poles') args.poles = Number(argv[++i]);
  else if (a === '--seed') args.seed = Number(argv[++i]);
  else if (a === '--random') args.random = true;
  else if (a === '--stochastic') args.greedy = false;
  else if (a === '--trace-only') args.panel = 'trace';
  else if (!a.startsWith('--')) args.policy = a;
}

// --- run an episode, sampling frames -------------------------------------
const env = new RunningEnv({ seed: args.seed, randomize: false });
const policy = args.policy ? Policy.fromJSON(JSON.parse(readFileSync(resolve(root, args.policy), 'utf8'))) : null;
const action = new Float64Array(ACTION_SIZE);

const sampleEvery = Math.max(1, Math.round(1 / (args.fps * CONTROL_DT)));
const maxSteps = Math.round(args.seconds / CONTROL_DT);
const frames = [];
const speeds = [];
const trunkTrace = [];
let reward = 0;
let steps = 0;
let fell = false;
let contactFrames = 0;
let travelled = 0;
let peakSpeed = 0;
let falls = 0;

while (steps < maxSteps) {
  if (policy && args.greedy) action.set(policy.actDeterministic(env.obs));
  else if (policy) action.set(policy.act(Float64Array.from(env.obs), Math.random).action);
  else for (let i = 0; i < ACTION_SIZE; i++) action[i] = Math.sin(steps * 0.03 + i * 1.1) * 0.7;
  const { reward: r, terminated, truncated } = env.step(action);
  reward += r;
  speeds.push(env.trunk.vx);
  peakSpeed = Math.max(peakSpeed, env.trunk.vx);
  // The environment resets in place when an episode ends, so the distance has to
  // be sampled while the episode is still live, not read from env afterwards.
  travelled = Math.max(travelled, env.trunk.x);
  trunkTrace.push([env.trunk.x, env.trunk.y]);
  const contacts = env.footContacts();
  if (contacts[0] || contacts[1]) contactFrames++;
  if (steps % sampleEvery === 0) frames.push({ snapshot: env.snapshot(), t: env.time, contacts });
  steps++;
  if (terminated || truncated) {
    fell = terminated;
    if (terminated) falls++;
    break;
  }
}

const meanSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
const duration = steps * CONTROL_DT;
const contactFraction = contactFrames / steps;

// --- figure layout --------------------------------------------------------
const pad = 20;
const robotHeight = 1.62; // trunk top when upright, metres
const topH = args.panel === 'trace' ? 0 : 380;
const bottomH = 360;
const width = args.width;
const height = (topH ? topH + 8 : 0) + bottomH;
const bmp = new Bitmap(width, height, [13, 16, 24]);

/** Stroboscopic trace panel: poses at their true world positions. */
function drawTrace(y0, panelH) {
  const xMin = -0.7;
  const xMax = Math.max(4.5, travelled + 1.3);
  const scale = Math.min((width - pad * 2) / (xMax - xMin), (panelH - pad * 2) / robotHeight);
  const project = (x, y) => [pad + (x - xMin) * scale, y0 + panelH - pad - y * scale];
  const [, groundY] = project(0, 0);
  const [gx0] = project(xMin, 0);
  const [gx1] = project(xMax, 0);
  bmp.fillRect(gx0, groundY, gx1, groundY + 1, COLORS.ground);
  for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
    const [px] = project(x, 0);
    const metre = Math.round(x);
    const long = metre % 5 === 0 ? 10 : 6;
    bmp.fillRect(px, groundY, px, groundY + long, metre % 5 === 0 ? [120, 136, 160] : COLORS.grid);
  }
  // trunk trajectory
  for (let i = 1; i < trunkTrace.length; i += 2) {
    const [x0, y0p] = project(trunkTrace[i - 1][0], trunkTrace[i - 1][1]);
    const [x1, y1p] = project(trunkTrace[i][0], trunkTrace[i][1]);
    bmp.line(x0, y0p, x1, y1p, [70, 96, 140], 0.85);
  }
  // Subsample: the point of this panel is the spacing between poses, so adjacent
  // silhouettes must not overlap into a blur.
  const stride = Math.max(1, Math.floor(frames.length / Math.min(9, frames.length)));
  const picked = frames.filter((_, i) => i % stride === 0);
  picked.forEach((frame, i) => {
    const age = picked.length > 1 ? i / (picked.length - 1) : 1;
    drawSnapshot(bmp, frame.snapshot, project, { alpha: 0.3 + 0.6 * age, shade: 0.45 + 0.55 * age, scale });
    for (let l = 0; l < frame.contacts.length; l++) {
      if (!frame.contacts[l]) continue;
      const [cx, cy] = project(frame.snapshot.x, 0);
      bmp.circle(cx, cy, 3, l === 0 ? [120, 90, 40] : [240, 180, 41], 0.9);
    }
  });
  console.log(`      trace panel: ${picked.length} poses, every ${(stride / args.fps).toFixed(2)} s`);
  return scale;
}

/** Flipbook panel: every pose drawn at the same pitch, like a gait plate. */
function drawFlipbook(panelH) {
  const poles = args.poles;
  const pitch = (width - pad * 2) / poles;
  const scale = Math.min(pitch / 1.15, (panelH - pad * 2) / robotHeight);
  const groundY = panelH - pad - 6;
  bmp.fillRect(pad, groundY, width - pad, groundY + 1, COLORS.ground);
  const stride = Math.max(1, Math.floor(frames.length / poles));
  for (let p = 0, i = 0; p < poles && i < frames.length; p++, i += stride) {
    const [cx0, cy0] = [pad + pitch * (p + 0.5), groundY];
    const project = (x, y) => [
      cx0 + (x - frames[i].snapshot.x) * scale,
      cy0 - y * scale,
    ];
    const age = frames.length > 1 ? i / (frames.length - 1) : 1;
    drawSnapshot(bmp, frames[i].snapshot, project, { alpha: 0.55 + 0.45 * age, shade: 0.62 + 0.38 * age, scale });
    if (p % 4 === 0) {
      bmp.fillRect(cx0 - pitch * 0.5, groundY + 6, cx0 - pitch * 0.5, groundY + 12, [70, 80, 96]);
    }
  }
  return scale;
}

if (topH) drawFlipbook(topH);
const traceScale = drawTrace(topH ? topH + 8 : 0, bottomH);

const outPath = resolve(root, args.out);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, bmp.toPNG());

console.log(`rendered ${frames.length} frames -> ${args.out} (flipbook scale ${traceScale.toFixed(0)} px/m)`);
console.log(
  `distance ${travelled.toFixed(2)} m | peak speed ${peakSpeed.toFixed(2)} m/s | mean speed ${meanSpeed.toFixed(3)} m/s (${(meanSpeed * 3.6).toFixed(2)} km/h) | ` +
    `duration ${duration.toFixed(2)} s | fell=${fell} | ground contact ${(contactFraction * 100).toFixed(0)}% of the time | reward ${reward.toFixed(2)}`,
);
console.log(`joint ranges: ${JOINTS.map((j) => `${j.name}[${j.min.toFixed(2)},${j.max.toFixed(2)}]`).join(' ')}`);
