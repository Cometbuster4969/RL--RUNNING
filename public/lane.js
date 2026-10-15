/**
 * A lane: a Web Worker that owns a PPO learner with its own environment, plus a
 * "showcase" environment that replays the learner's current policy at a
 * watchable pace.
 *
 * Keeping those two apart is what makes the page both honest and watchable: the
 * robot moving on the track really is the lane's current policy, but the learner
 * behind it does not have to be slowed to human speed for you to see it.
 *
 * This worker imports exactly the same modules as the Node trainer and the tests.
 */
import { PPOTrainer } from '../src/ppo.js';
import { RunningEnv, CONTROL_DT } from '../src/env.js';

const state = {
  running: false,
  turbo: true,
  speed: 2,
  laneId: 0,
  name: 'lane',
  color: '#4c9bff',
  trainer: null,
  showcase: null,
  showAction: null,
  episodeLog: [],
  epTime: 0,
  epDist: 0,
  showEpisode: 1,
  showFalls: 0,
  lastHistoryEpisodes: 0,
  maxSteps: Infinity,
  timer: null,
  history: [],
};

function mulberryish(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function init(msg) {
  state.laneId = msg.laneId ?? state.laneId;
  state.name = msg.name ?? state.name;
  state.color = msg.color ?? state.color;
  state.speed = msg.speed ?? state.speed;
  state.turbo = msg.turbo !== undefined ? msg.turbo : state.turbo;
  state.maxSteps = msg.maxSteps ?? Infinity;

  state.trainer = new PPOTrainer({
    seed: msg.seed ?? 1,
    hidden: [64, 64],
    policyJSON: msg.policyJSON || null,
    policyPerturb: msg.policyPerturb || 0,
    envOpts: { randomize: true },
  });
  if (!msg.policyJSON) {
    // Distinct starting weights per lane, so the lanes do not move in lockstep.
    const rng = mulberryish((msg.seed ?? 1) * 1013 + 7);
    const { params } = state.trainer.policy.collect();
    for (const p of params) {
      for (let i = 0; i < p.length; i++) {
        p[i] += (rng() * 2 - 1) * 0.05 * (Math.abs(p[i]) + 0.05);
      }
    }
    for (let i = 0; i < state.trainer.policy.logStd.length; i++) {
      state.trainer.policy.logStd[i] = -0.7 + (rng() * 2 - 1) * 0.35;
    }
  }

  state.showcase = new RunningEnv({ seed: (msg.seed ?? 1) * 37 + 11, randomize: false });
  state.showAction = new Float64Array(state.trainer.actSize);
  state.history = [];
  state.episodeLog = [];
  state.epTime = 0;
  state.epDist = 0;
  state.showEpisode = 1;
  state.showFalls = 0;
  state.lastHistoryEpisodes = 0;

  post({ type: 'ready', laneId: state.laneId });
  postStatus();
  schedule(state.running ? 0 : 120);
}

function post(message) {
  self.postMessage(message);
}

/** Advance the visible rollout by `controlSteps` control steps. */
function stepShowcase(controlSteps) {
  const env = state.showcase;
  const policy = state.trainer.policy;
  for (let i = 0; i < controlSteps; i++) {
    state.showAction.set(policy.actDeterministic(env.obs));
    const { terminated, truncated } = env.step(state.showAction);
    state.epTime += CONTROL_DT;
    state.epDist = Math.max(state.epDist, env.trunk.x);
    if (terminated || truncated) {
      state.showEpisode++;
      if (terminated) state.showFalls++;
      state.episodeLog.push({ distance: round2(state.epDist), time: round2(state.epTime), fell: terminated });
      if (state.episodeLog.length > 200) state.episodeLog.shift();
      state.epTime = 0;
      state.epDist = 0;
      env.reset();
    }
  }
}

function round2(v) {
  return Math.round(v * 100) / 100;
}

/** Evenly subsample the episode history, always keeping the last point. */
function downsample(history, maxPoints) {
  if (history.length <= maxPoints) return history.map((e) => [e[0], e[1]]);
  const stride = history.length / maxPoints;
  const out = [];
  for (let k = 0; k < maxPoints; k++) {
    const e = history[Math.floor(k * stride)];
    out.push([e[0], e[1]]);
  }
  const last = history[history.length - 1];
  out.push([last[0], last[1]]);
  return out;
}

/** One slice of work: animate, learn, report. */
function slice() {
  const sliceMs = state.turbo ? 100 : 100;
  const t0 = performance.now();

  // 1. The visible rollout: wall-clock time times the animation speed.
  if (state.speed > 0) {
    const controlSteps = Math.max(1, Math.round((sliceMs / 1000) * 60 * state.speed));
    stepShowcase(controlSteps);
  }

  // 2. Training. Turbo: run the learner flat out for the rest of the slice.
  if (state.running && state.trainer.steps < state.maxSteps) {
    const budgetMs = state.turbo ? Math.max(5, sliceMs - (performance.now() - t0) - 4) : 2;
    const t1 = performance.now();
    while (performance.now() - t1 < budgetMs && state.trainer.steps < state.maxSteps) {
      state.trainer.trainSteps(128);
    }
    // Append new episode outcomes to the chart history.
    const h = state.trainer.history;
    while (state.history.length < h.length) {
      const e = h[state.history.length];
      state.history.push([e.step, round2(e.distance), e.time, e.reward]);
    }
  }

  postStatus();
  const elapsed = performance.now() - t0;
  schedule(state.running ? Math.max(0, sliceMs - elapsed) : 120);
}

function schedule(delay) {
  if (state.timer !== null) clearTimeout(state.timer);
  state.timer = setTimeout(slice, delay);
}

function postStatus() {
  const trainer = state.trainer;
  const env = state.showcase;
  const snapshot = env.snapshot();
  // Pace over the current episode (not the instantaneous velocity, which is noisy).
  const pace = state.epTime > 0.4 ? state.epDist / state.epTime : Math.max(0, env.trunk.vx);
  const stats = trainer.stats();
  post({
    type: 'status',
    laneId: state.laneId,
    name: state.name,
    color: state.color,
    steps: stats.steps,
    updates: stats.updates,
    episodes: stats.episodes,
    best: stats.best,
    recent: stats.recent,
    kl: stats.kl,
    entropy: stats.entropy,
    valueLoss: stats.valueLoss,
    params: trainer.policy.paramCount(),
    showcase: {
      x: snapshot.x,
      xAbs: snapshot.totalX,
      t: state.epTime,
      distance: round2(state.epDist),
      pace: round2(pace * 100) / 100,
      kmh: Math.round(pace * 3.6 * 10) / 10,
      episode: state.showEpisode,
      falls: state.showFalls,
      episodeLog: state.episodeLog.slice(-12),
      snapshot,
    },
    learning: {
      episode: stats.env.distance,
      time: stats.env.time,
      index: trainer.env.episode,
      completed: stats.episodes,
    },
    // Downsampled full curve: the chart must show the whole run, not a tail.
    history: downsample(state.history, 600),
    finished: trainer.steps >= state.maxSteps,
  });
}

self.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      init(msg);
      break;
    case 'start':
      state.running = true;
      if (msg.maxSteps) state.maxSteps = msg.maxSteps;
      schedule(0);
      break;
    case 'pause':
      state.running = false;
      break;
    case 'turbo':
      state.turbo = !!msg.value;
      break;
    case 'speed':
      state.speed = msg.value;
      break;
    case 'reset':
      init({ ...msg, seed: msg.seed ?? 1 + state.laneId * 101 });
      if (msg.andStart) {
        state.running = true;
        schedule(0);
      }
      break;
    case 'checkpoint':
      post({
        type: 'checkpoint',
        laneId: state.laneId,
        name: state.name,
        steps: state.trainer.steps,
        best: state.trainer.bestDistance,
        policy: state.trainer.policy.toJSON(),
      });
      break;
    case 'stop':
      state.running = false;
      if (state.timer !== null) clearTimeout(state.timer);
      self.close();
      break;
    default:
      break;
  }
};
