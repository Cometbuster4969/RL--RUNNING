/**
 * Evaluate a saved policy: run it over a batch of episodes and report what it
 * can actually do. Deterministic (greedy) unless you ask for --stochastic.
 *
 *   node scripts/play.mjs runs/champion/lane1-best.json --episodes 20
 *   node scripts/play.mjs runs/champion/lane1-best.json --episodes 20 --render runs/play.png
 */
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Policy } from '../src/net.js';
import { evaluatePolicy, formatEvaluation } from '../src/evaluate.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const args = { policy: null, episodes: 10, seed: 1234, stochastic: false, maxSeconds: 60, render: null };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--episodes') args.episodes = Number(argv[++i]);
  else if (a === '--seed') args.seed = Number(argv[++i]);
  else if (a === '--stochastic') args.stochastic = true;
  else if (a === '--max-seconds') args.maxSeconds = Number(argv[++i]);
  else if (a === '--render') args.render = argv[++i];
  else if (!a.startsWith('--')) args.policy = a;
}

if (!args.policy) {
  console.error('usage: node scripts/play.mjs <policy.json> [--episodes N] [--stochastic] [--render out.png]');
  process.exit(1);
}

const policy = Policy.fromJSON(JSON.parse(readFileSync(resolve(root, args.policy), 'utf8')));
const stats = evaluatePolicy(policy, {
  episodes: args.episodes,
  seed: args.seed,
  stochastic: args.stochastic,
  maxSeconds: args.maxSeconds,
});

console.log('');
console.log(formatEvaluation(stats));
console.log('');

if (args.render) {
  // Reuse the film-strip renderer for a single-episode figure.
  const { spawnSync } = await import('node:child_process');
  spawnSync(process.execPath, [join(here, 'gait-strip.mjs'), args.policy, '--out', args.render, '--seconds', '8'], {
    stdio: 'inherit',
    cwd: root,
  });
}
