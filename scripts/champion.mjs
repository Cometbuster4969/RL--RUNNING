/**
 * Produce (or refresh) the reference champion policy shipped in runs/champion.
 *
 *   node scripts/champion.mjs                       # rank whatever is in runs/champion
 *   node scripts/champion.mjs --train --minutes 8 --lanes 4
 *   node scripts/champion.mjs --dir runs/some-run --episodes 16
 *
 * The champion is not "the checkpoint with the best training number" - training
 * numbers come from the exploration policy and are noisy. Instead every
 * checkpoint in the directory is evaluated on the same fixed set of starting
 * conditions and ranked by `scoreOf` (distance, speed, and a heavy penalty for
 * falling), then the winner is written as `champion.json` with a figure and a
 * metadata file.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, readdirSync, existsSync, statSync, mkdirSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Policy } from '../src/net.js';
import { evaluatePolicy, formatEvaluation } from '../src/evaluate.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const args = {
  /** Directories to collect candidate checkpoints from (may be repeated). */
  dirs: [],
  out: null,
  outName: 'champion.json',
  episodes: 12,
  seed: 4242,
  train: false,
  minutes: 8,
  lanes: 4,
  steps: 0,
  force: false,
  quiet: false,
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--dir') args.dirs.push(argv[++i]);
  else if (a === '--out') args.out = argv[++i];
  else if (a === '--episodes') args.episodes = Number(argv[++i]);
  else if (a === '--seed') args.seed = Number(argv[++i]);
  else if (a === '--train') args.train = true;
  else if (a === '--minutes') args.minutes = Number(argv[++i]);
  else if (a === '--steps') args.steps = Number(argv[++i]);
  else if (a === '--lanes') args.lanes = Number(argv[++i]);
  else if (a === '--force') args.force = true;
  else if (a === '--quiet') args.quiet = true;
}

if (!args.dirs.length) args.dirs.push('runs/champion');
const dirs = args.dirs.map((d) => resolve(root, d));
const outDir = resolve(root, args.out ?? args.dirs[0]);
const championPath = join(outDir, args.outName);

if (existsSync(championPath) && !args.force && !args.train) {
  const meta = existsSync(join(outDir, 'champion.meta.json'))
    ? JSON.parse(readFileSync(join(outDir, 'champion.meta.json'), 'utf8'))
    : null;
  console.log(`champion already exists: ${championPath}`);
  if (meta) {
    console.log(`  steps trained: ${meta.steps}`);
    console.log(`  greedy score:  ${meta.score?.toFixed?.(3) ?? '?'}`);
    console.log(`  median distance: ${meta.medianDistance} m, falls ${(meta.fallRate * 100).toFixed(0)}%`);
    console.log(`  produced: ${meta.producedAt}`);
  }
  console.log('\nuse --force to re-rank, or --train to train a fresh batch first.');
  process.exit(0);
}

// ---------------------------------------------------------------- training
if (args.train) {
  const trainArgs = [
    join(here, 'train.mjs'),
    '--lanes',
    String(args.lanes),
    '--out',
    args.dirs[0],
    '--eval-every',
    '250000',
  ];
  trainArgs[3] = args.dirs[0];
  if (args.steps > 0) trainArgs.push('--steps', String(args.steps));
  else trainArgs.push('--minutes', String(args.minutes));
  console.log(`\ntraining: node scripts/train.mjs ${trainArgs.slice(1).join(' ')}\n`);
  const result = spawnSync(process.execPath, trainArgs, { stdio: 'inherit', cwd: root });
  if (result.status !== 0) {
    console.error('training failed');
    process.exit(result.status ?? 1);
  }
}

// -------------------------------------------------------------- candidates
const candidates = [];
for (const dir of dirs) {
  if (!existsSync(dir)) {
    console.error(`no such directory: ${dir}`);
    process.exit(1);
  }
  for (const file of readdirSync(dir)) {
    if (!file.endsWith('.json')) continue;
    if (file.endsWith('.meta.json') || file.endsWith('-history.json')) continue;
    if (file.startsWith('champion') || file === 'run.json') continue;
    const path = join(dir, file);
    const size = statSync(path).size;
    try {
      const parsed = JSON.parse(readFileSync(path, 'utf8'));
      if (parsed.format !== 'rl-running-policy@1') continue;
      candidates.push({ file, path, parsed, size, dir, label: `${basename(dir)}/${file}` });
    } catch {
      /* ignore unreadable files */
    }
  }
}

if (!candidates.length) {
  console.error(`no policy checkpoints found in ${args.dirs.join(', ')}`);
  process.exit(1);
}

console.log(`\nranking ${candidates.length} checkpoint(s) from ${args.dirs.join(', ')} over ${args.episodes} fixed episodes each\n`);
const header = 'checkpoint                 median   mean    best   m/s    km/h  falls   score';
console.log(header);
console.log('-'.repeat(header.length));

const ranked = [];
for (const candidate of candidates) {
  const policy = Policy.fromJSON(candidate.parsed);
  const stats = evaluatePolicy(policy, { episodes: args.episodes, seed: args.seed });
  // A second, sampled pass: an honest runner should survive exploration noise too.
  const sampled = evaluatePolicy(policy, { episodes: Math.max(3, Math.round(args.episodes / 3)), seed: args.seed + 5, stochastic: true });
  const combined = stats.score * 0.75 + sampled.score * 0.25;
  ranked.push({ ...candidate, policy, stats, sampled, combined });
  console.log(
    `${candidate.label.padEnd(30)} ${stats.medianDistance.toFixed(2).padStart(7)} ${stats.meanDistance.toFixed(2).padStart(7)} ${stats.bestDistance.toFixed(2).padStart(7)} ${stats.meanSpeed.toFixed(2).padStart(5)} ${(stats.meanSpeed * 3.6).toFixed(1).padStart(6)} ${(stats.fallRate * 100).toFixed(0).padStart(5)}% ${combined.toFixed(2).padStart(7)}`,
  );
}

ranked.sort((a, b) => b.combined - a.combined);
const best = ranked[0];

console.log(`\nwinner: ${best.file}\n  ${formatEvaluation(best.stats, 'greedy')}`);
console.log(`  ${formatEvaluation(best.sampled, 'sampled')}`);

mkdirSync(outDir, { recursive: true });
writeFileSync(championPath, JSON.stringify(best.policy.toJSON()));
writeFileSync(
  join(outDir, 'champion.meta.json'),
  JSON.stringify(
    {
      source: best.label,
      sourcePath: relative(root, best.path),
      sourceRun: relative(root, best.dir),
      steps: readSteps(best.dir, best.file),
      score: best.combined,
      greedy: best.stats,
      sampled: best.sampled,
      producedAt: new Date().toISOString(),
      args,
    },
    null,
    2,
  ),
);

// Figure + learning curves, if the plotter is available.
if (!args.quiet) {
  console.log('\nrendering figure...');
  spawnSync(
    process.execPath,
    [join(here, 'gait-strip.mjs'), championPath, '--out', join(outDir, 'gait-champion.png'), '--seconds', '8'],
    { stdio: 'inherit', cwd: root },
  );
  // The learning curve comes from the run the winner was found in, so the figure
  // always describes the policy that is actually shipped.
  spawnSync(process.execPath, [join(here, 'chart.mjs'), best.dir, '--out', join(outDir, 'curves.png')], {
    stdio: 'inherit',
    cwd: root,
  });
}

console.log(`\nwrote ${relative(root, championPath)}`);
console.log('the browser can now load it with the "Load champion" button.');

/** Steps trained for a checkpoint, from its run directory's run.json. */
function readSteps(dir, file) {
  try {
    const run = JSON.parse(readFileSync(join(dir, 'run.json'), 'utf8'));
    const lane = run.lanes?.find?.((entry) => file.startsWith(`lane${entry.lane}`));
    return lane?.steps ?? null;
  } catch {
    return null;
  }
}
