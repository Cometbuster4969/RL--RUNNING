/**
 * Throughput benchmark: how many experience steps per second can one lane
 * collect, and how much does the PPO update cost on top of that?
 *
 *   node scripts/bench.mjs [steps] [lanes]
 */
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { RunningEnv, OBS_SIZE, ACTION_SIZE, CONTROL_DT } from '../src/env.js';
import { Policy } from '../src/net.js';
import { PPOTrainer } from '../src/ppo.js';
import { mulberry32, formatCount } from '../src/math.js';

const here = dirname(fileURLToPath(import.meta.url));
const steps = Number(process.argv[2] ?? 20000);
const lanes = Number(process.argv[3] ?? 1);

function line(label, value) {
  console.log(`${label.padEnd(46)} ${value}`);
}

console.log('\n=== RL--RUNNING benchmark ===\n');

// 1. Environment only ------------------------------------------------------
{
  const env = new RunningEnv({ seed: 1 });
  const action = new Float64Array(ACTION_SIZE);
  for (let i = 0; i < action.length; i++) action[i] = Math.sin(i * 1.7) * 0.7;
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) env.step(action);
  const ms = performance.now() - t0;
  line('environment steps/s (1 core)', formatCount((steps / ms) * 1000));
  line('  = realtime factor', `${((steps * CONTROL_DT) / (ms / 1000)).toFixed(0)}x`);
  line('  = simulated seconds per wall second', formatCount((steps * CONTROL_DT) / (ms / 1000)));
}

// 2. Policy forward pass only ---------------------------------------------
{
  const policy = new Policy({ obsSize: OBS_SIZE, actSize: ACTION_SIZE, hidden: [64, 64], seed: 1 });
  const rng = mulberry32(3);
  const obs = new Float64Array(OBS_SIZE).map(() => Math.random() - 0.5);
  const t0 = performance.now();
  const N = 200000;
  for (let i = 0; i < N; i++) policy.act(obs, rng);
  const ms = performance.now() - t0;
  line('policy sampling (act) per second', formatCount((N / ms) * 1000));
}

// 3. Full training loop (env + rollout + PPO update), single lane ----------
{
  const trainer = new PPOTrainer({ seed: 1, hidden: [64, 64] });
  const t0 = performance.now();
  trainer.trainSteps(steps);
  const ms = performance.now() - t0;
  line('full PPO training steps/s (1 lane, 1 core)', formatCount((steps / ms) * 1000));
  line('  = updates', String(trainer.updates));
  line('  = episodes completed', String(trainer.episodes));
  line('  = mean distance over those episodes', (trainer.history.reduce((a, h) => a + h.distance, 0) / Math.max(1, trainer.history.length)).toFixed(3));
  line('  = policy parameters', String(trainer.policy.paramCount()));
  line('  = approx KL after last update', trainer.kl.toExponential(2));
}

// 4. Parallel lanes, as used by the server / trainer ----------------------
if (lanes > 1) {
  await new Promise((resolve) => {
    const workers = [];
    const perLane = Math.max(1000, Math.round(steps / lanes));
    let ready = 0;
    let done = 0;
    const t0 = performance.now();
    for (let i = 0; i < lanes; i++) {
      const w = new Worker(join(here, 'lane-worker.mjs'), {
        workerData: { laneId: i, name: `lane-${i}`, seed: i + 1, hidden: [64, 64], hyper: {}, sliceMs: 1e9 },
      });
      workers.push(w);
      w.on('message', (msg) => {
        if (msg.type === 'ready') {
          ready++;
          if (ready === lanes) for (const ww of workers) ww.postMessage({ type: 'start', steps: perLane });
        }
        if (msg.type === 'done') {
          done++;
          if (done === lanes) {
            const ms = performance.now() - t0;
            const total = perLane * lanes;
            line(`${lanes} parallel lanes, aggregate steps/s`, formatCount((total / ms) * 1000));
            line('  = steps/s per lane', formatCount((perLane / ms) * 1000));
            for (const ww of workers) ww.postMessage({ type: 'stop' });
            resolve();
          }
        }
      });
    }
  });
}

console.log('');
