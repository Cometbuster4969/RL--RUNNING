/**
 * Headless training: run one or more independent lanes (worker threads) and log
 * their progress. Checkpoints are written as policy JSON so they can be loaded
 * by the browser race or by `scripts/play.mjs`.
 *
 *   node scripts/train.mjs --lanes 2 --steps 2000000 --out runs/champion
 *   node scripts/train.mjs --minutes 10 --lanes 2 --load runs/champion/best.json
 */
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { formatCount, formatDuration } from '../src/math.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

function parseArgs(argv) {
  const args = { lanes: 1, steps: 1_000_000, minutes: 0, out: 'runs/default', seed: 1, evalEvery: 100_000, hidden: [64, 64], perturb: 0.02 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--lanes') args.lanes = Number(next());
    else if (a === '--steps') args.steps = Number(next());
    else if (a === '--minutes') args.minutes = Number(next());
    else if (a === '--out') args.out = next();
    else if (a === '--seed') args.seed = Number(next());
    else if (a === '--eval-every') args.evalEvery = Number(next());
    else if (a === '--load') args.load = next();
    else if (a === '--perturb') args.perturb = Number(next());
    else if (a === '--hidden') args.hidden = next().split(',').map(Number);
    else if (a === '--quiet') args.quiet = true;
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const outDir = resolve(root, args.out);
mkdirSync(outDir, { recursive: true });

let loadedPolicy = null;
if (args.load) {
  const path = resolve(root, args.load);
  if (!existsSync(path)) {
    console.error(`cannot load checkpoint: ${path} does not exist`);
    process.exit(1);
  }
  loadedPolicy = JSON.parse(readFileSync(path, 'utf8'));
  console.log(`loaded checkpoint ${args.load} (${loadedPolicy.hidden.join('x')} hidden, ${loadedPolicy.obsSize}->${loadedPolicy.actSize})`);
}

const deadline = args.minutes > 0 ? Date.now() + args.minutes * 60_000 : Infinity;
const lanes = [];
const state = new Map();
let finished = 0;

console.log(`\nstarted ${args.lanes} lane(s)  target ${formatCount(args.steps)} steps each  writing to ${args.out}\n`);
const header = 'lane      steps   updates    eps   best(m)  recent(m)   km/h   kl       eps/s';
console.log(header);
console.log('-'.repeat(header.length));

for (let i = 0; i < args.lanes; i++) {
  const worker = new Worker(join(here, 'lane-worker.mjs'), {
    workerData: {
      laneId: i,
      name: `lane-${i}`,
      seed: args.seed + i * 101,
      hidden: args.hidden,
      hyper: {},
      policyJSON: loadedPolicy,
      policyPerturb: loadedPolicy ? args.perturb * (i + 1) : 0,
      sliceMs: 400,
    },
  });
  lanes.push(worker);
  state.set(i, { steps: 0, start: Date.now(), lastEval: 0, bestEval: 0, done: false, best: 0 });

  worker.on('message', (msg) => {
    if (msg.type === 'ready') {
      worker.postMessage({ type: 'start', steps: args.steps });
      return;
    }
    if (msg.type === 'stats') {
      const s = state.get(i);
      s.steps = msg.steps;
      s.episodes = msg.episodes;
      s.best = msg.best;
      s.recent = msg.recent;
      s.ema = msg.ema;
      s.kl = msg.kl;
      s.curve = msg.curve;
      const elapsed = (Date.now() - s.start) / 1000;
      const kmh = msg.recent > 0 ? (msg.recent / Math.max(0.5, msg.snapshot.t)) * 3.6 : 0;
      if (!args.quiet) {
        console.log(
          `${msg.name.padEnd(6)} ${formatCount(msg.steps).padStart(9)} ${String(msg.updates).padStart(9)} ${String(msg.episodes).padStart(6)} ${msg.best.toFixed(2).padStart(8)} ${msg.recent.toFixed(3).padStart(10)} ${kmh.toFixed(1).padStart(6)} ${msg.kl.toExponential(1).padStart(8)} ${(msg.episodes / elapsed).toFixed(1).padStart(7)}`,
        );
      }
      // Keep a partial history on disk so a run can be plotted or killed at any
      // time and still produce a learning curve.
      s.historyWrites = (s.historyWrites ?? 0) + 1;
      if (s.historyWrites % 5 === 0 && msg.curve) {
        writeFileSync(join(outDir, `lane${i}-history.json`), JSON.stringify(msg.curve));
      }
      // Ask for a greedy evaluation at intervals - the honest progress signal.
      if (msg.steps - s.lastEval >= args.evalEvery) {
        s.lastEval = msg.steps;
        worker.postMessage({ type: 'evaluate', episodes: 3 });
        worker.postMessage({ type: 'checkpoint' });
      }
      if (Date.now() > deadline) worker.postMessage({ type: 'pause' });
      return;
    }
    if (msg.type === 'evaluation') {
      const s = state.get(i);
      s.bestEval = Math.max(s.bestEval, msg.result.mean);
      console.log(
        `  >> lane-${i} greedy eval: mean ${msg.result.mean.toFixed(2)} m, best ${msg.result.best.toFixed(2)} m, fall rate ${(msg.result.fallRate * 100).toFixed(0)}%`,
      );
      return;
    }
    if (msg.type === 'checkpoint') {
      const s = state.get(i);
      const evalScore = s.bestEval;
      const fileName = msg.steps >= args.steps ? `lane${i}-final.json` : `lane${i}-latest.json`;
      writeFileSync(join(outDir, fileName), JSON.stringify(msg.policy));
      if (evalScore >= (s.savedEval ?? -1)) {
        s.savedEval = evalScore;
        writeFileSync(join(outDir, `lane${i}-best.json`), JSON.stringify(msg.policy));
        writeFileSync(
          join(outDir, `lane${i}-best.meta.json`),
          JSON.stringify({ steps: msg.steps, greedyEvalMean: evalScore, trainedAt: new Date().toISOString(), args }, null, 2),
        );
      }
      return;
    }
    if (msg.type === 'done') {
      finished++;
      const s = state.get(i);
      s.done = true;
      writeFileSync(join(outDir, `lane${i}-final.json`), JSON.stringify(msg.policy || s.lastPolicy || {}));
      writeFileSync(join(outDir, `lane${i}-history.json`), JSON.stringify(msg.curve, null, 0));
      console.log(`lane-${i} finished ${formatCount(msg.steps)} steps, best training distance ${msg.best.toFixed(2)} m`);
      if (finished === args.lanes) finish();
    }
  });

  worker.on('error', (err) => {
    console.error(`lane-${i} crashed:`, err);
  });
}

function finish() {
  const history = [];
  for (const [i, s] of state) {
    history.push({ lane: i, steps: s.steps, best: s.best, recent: s.recent, bestEval: s.bestEval, curve: s.curve || [] });
  }
  writeFileSync(join(outDir, 'run.json'), JSON.stringify({ args, finishedAt: new Date().toISOString(), lanes: history }, null, 2));
  const totalSteps = history.reduce((a, h) => a + h.steps, 0);
  const seconds = (Date.now() - Math.min(...[...state.values()].map((s) => s.start))) / 1000;
  console.log(
    `\ndone: ${formatCount(totalSteps)} steps in ${formatDuration(seconds)} (${formatCount(totalSteps / seconds)} steps/s aggregate)`,
  );
  for (const h of history) {
    console.log(`  lane-${h.lane}: best training distance ${h.best.toFixed(2)} m, best greedy eval ${h.bestEval.toFixed(2)} m`);
  }
  for (const w of lanes) w.postMessage({ type: 'stop' });
  setTimeout(() => process.exit(0), 200);
}

if (args.minutes > 0) {
  const timer = setInterval(() => {
    if (Date.now() > deadline) {
      clearInterval(timer);
      for (const w of lanes) w.postMessage({ type: 'pause' });
      finish();
    }
  }, 1000);
}

process.on('SIGINT', () => {
  console.log('\ninterrupted - saving checkpoints');
  for (const w of lanes) w.postMessage({ type: 'checkpoint' });
  setTimeout(finish, 800);
});
