/**
 * One training "lane", as a worker thread.
 *
 * The main thread owns the UI / API, the worker owns a single agent: its own
 * environment, its own policy and its own optimiser. That is the same structure
 * the browser race uses - each lane is an independent learner, and the only
 * thing they share is the wall clock.
 */
import { parentPort, workerData } from 'node:worker_threads';
import { createLane as buildLane } from '../src/ppo.js';
import { Policy } from '../src/net.js';

const lane = buildLane({
  seed: workerData.seed,
  hidden: workerData.hidden,
  hyper: workerData.hyper,
  policyJSON: workerData.policyJSON,
  policyPerturb: workerData.policyPerturb,
  envOpts: { randomize: true },
});

if (!workerData.policyJSON) {
  // Make sure each lane still starts from a different initial policy.
  const { params } = lane.policy.collect();
  let k = 0;
  for (const p of params) {
    for (let i = 0; i < p.length; i++) {
      p[i] += Math.sin(++k * 12.9898 + workerData.seed * 78.233) * 0.01 * (Math.abs(p[i]) + 0.05);
    }
  }
}

let running = false;
let targetSteps = Infinity;
const sliceMs = workerData.sliceMs ?? 150;

function post(message) {
  parentPort.postMessage(message);
}

/** Evenly subsample a history array, always keeping the last point. */
function downsample(history, maxPoints) {
  if (history.length <= maxPoints) {
    return history.map((h) => [h.step, Math.round(h.distance * 100) / 100]);
  }
  const stride = history.length / maxPoints;
  const out = [];
  for (let k = 0; k < maxPoints; k++) {
    const h = history[Math.floor(k * stride)];
    out.push([h.step, Math.round(h.distance * 100) / 100]);
  }
  const last = history[history.length - 1];
  out.push([last.step, Math.round(last.distance * 100) / 100]);
  return out;
}

function status() {
  const s = lane.stats();
  return {
    type: 'stats',
    laneId: workerData.laneId,
    name: workerData.name,
    steps: s.steps,
    updates: s.updates,
    episodes: s.episodes,
    best: s.best,
    recent: s.recent,
    ema: s.ema,
    kl: s.kl,
    entropy: s.entropy,
    valueLoss: s.valueLoss,
    policyLoss: s.policyLoss,
    params: lane.policy.paramCount(),
    snapshot: lane.env.snapshot(),
    episode: {
      distance: s.env.distance,
      time: s.env.time,
      index: lane.env.episode,
    },
    // The full curve, downsampled to ~600 points: enough to plot the whole run
    // without shipping tens of thousands of samples on every status message.
    curve: downsample(lane.history, 600),
  };
}

async function loop() {
  while (running && lane.steps < targetSteps) {
    const t0 = Date.now();
    while (Date.now() - t0 < sliceMs && lane.steps < targetSteps) {
      lane.trainSteps(256);
    }
    post(status());
    // Yield so control messages (pause / save / evaluate) get processed.
    await new Promise((resolve) => setImmediate(resolve));
  }
  if (lane.steps >= targetSteps) {
    post({ ...status(), type: 'done' });
    running = false;
  }
}

parentPort.on('message', (msg) => {
  switch (msg.type) {
    case 'start': {
      targetSteps = msg.steps ?? Infinity;
      if (!running) {
        running = true;
        loop();
      }
      break;
    }
    case 'pause':
      running = false;
      post({ ...status(), type: 'paused' });
      break;
    case 'reset': {
      const stopped = !running;
      running = false;
      lane.env.reset();
      lane.history.length = 0;
      lane.steps = 0;
      lane.updates = 0;
      lane.episodes = 0;
      lane.bestDistance = 0;
      lane.distanceEma = 0;
      if (!stopped) {
        running = true;
        loop();
      }
      post(status());
      break;
    }
    case 'evaluate': {
      const result = lane.evaluate(msg.episodes ?? 3, { greedy: true });
      post({ type: 'evaluation', laneId: workerData.laneId, result });
      break;
    }
    case 'checkpoint': {
      post({ type: 'checkpoint', laneId: workerData.laneId, steps: lane.steps, policy: lane.policy.toJSON(), stats: lane.stats() });
      break;
    }
    case 'load': {
      const stopped = !running;
      running = false;
      lane.policy.copyFrom(Policy.fromJSON(msg.policy));
      if (!stopped) {
        running = true;
        loop();
      }
      post(status());
      break;
    }
    case 'stop':
      running = false;
      parentPort.close();
      break;
    default:
      break;
  }
});

post({ type: 'ready', laneId: workerData.laneId, name: workerData.name, params: lane.policy.paramCount() });
