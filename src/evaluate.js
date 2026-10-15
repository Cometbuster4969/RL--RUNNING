/**
 * Shared policy evaluation, used by `scripts/play.mjs`, `scripts/champion.mjs`
 * and the test suite.
 *
 * Evaluation always uses fixed episode seeds, so two policies are compared on
 * exactly the same set of starting conditions and the comparison is fair and
 * reproducible.
 */
import { RunningEnv, CONTROL_DT } from './env.js';
import { mulberry32 } from './math.js';

/**
 * Run a policy for a batch of episodes.
 *
 * @param {import('./net.js').Policy} policy
 * @param {{episodes?:number, seed?:number, stochastic?:boolean, maxSeconds?:number,
 *          env?:RunningEnv}} opts
 */
export function evaluatePolicy(policy, opts = {}) {
  const episodes = opts.episodes ?? 10;
  const seed = opts.seed ?? 1234;
  const maxSeconds = opts.maxSeconds ?? 60;
  // The start pose is jittered (deterministically, from the environment's own
  // seeded RNG) so that the per-episode spread is informative. Because the RNG is
  // seeded, every policy still faces exactly the same set of starts.
  const env = opts.env || new RunningEnv({ seed: seed + 1, randomize: opts.startJitter ?? true });
  const rng = mulberry32(seed * 31 + 977);
  const action = new Float64Array(policy.actSize);
  const results = [];

  for (let e = 0; e < episodes; e++) {
    env.reset();
    let steps = 0;
    let distance = 0;
    let speedSum = 0;
    let peakSpeed = 0;
    let groundFrames = 0;
    let airFrames = 0;
    let done = false;
    while (!done && steps < maxSeconds / CONTROL_DT) {
      if (opts.stochastic) action.set(policy.act(Float64Array.from(env.obs), rng).action);
      else action.set(policy.actDeterministic(env.obs));
      const { terminated, truncated } = env.step(action);
      distance = Math.max(distance, env.trunk.x);
      speedSum += env.trunk.vx;
      peakSpeed = Math.max(peakSpeed, env.trunk.vx);
      const contacts = env.footContacts();
      if (contacts[0] || contacts[1]) groundFrames++;
      else airFrames++;
      steps++;
      done = terminated || truncated;
    }
    results.push({
      distance: Number(distance.toFixed(3)),
      time: Number((steps * CONTROL_DT).toFixed(3)),
      meanSpeed: Number((speedSum / steps).toFixed(3)),
      peakSpeed: Number(peakSpeed.toFixed(3)),
      groundContact: Number((groundFrames / steps).toFixed(3)),
      airborne: Number((airFrames / steps).toFixed(3)),
      fell: env.terminatedByFall,
    });
  }

  const distances = results.map((r) => r.distance).slice().sort((a, b) => a - b);
  const speeds = results.map((r) => r.meanSpeed).slice().sort((a, b) => a - b);
  const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const median = (arr) => (arr.length ? arr[Math.floor(arr.length / 2)] : 0);
  const fallRate = results.filter((r) => r.fell).length / results.length;

  const stats = {
    episodes: results.length,
    meanDistance: Number(mean(distances).toFixed(3)),
    medianDistance: Number(median(distances).toFixed(3)),
    bestDistance: Number((distances[distances.length - 1] ?? 0).toFixed(3)),
    meanSpeed: Number(mean(speeds).toFixed(3)),
    bestEpisodeSpeed: Number((speeds[speeds.length - 1] ?? 0).toFixed(3)),
    fallRate,
    groundContact: Number(mean(results.map((r) => r.groundContact)).toFixed(3)),
    airborne: Number(mean(results.map((r) => r.airborne)).toFixed(3)),
    results,
  };
  stats.score = scoreOf(stats);
  return stats;
}

/**
 * A single number for ranking policies.
 *
 * Distance is what we want, but a policy that sprints 20 m and then falls over
 * on a third of its attempts is not a runner: falling is penalised hard, and
 * there is a small bonus for sustained speed so that a slow, very stable shuffle
 * does not win by default.
 */
export function scoreOf(stats) {
  return stats.medianDistance + 2 * stats.meanSpeed - 25 * stats.fallRate;
}

/** Format an evaluation for the terminal. */
export function formatEvaluation(stats, label = '') {
  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  return [
    `${label ? label + ': ' : ''}${stats.episodes} episodes`,
    `distance median ${stats.medianDistance.toFixed(2)} m / mean ${stats.meanDistance.toFixed(2)} m / best ${stats.bestDistance.toFixed(2)} m`,
    `speed    ${stats.meanSpeed.toFixed(2)} m/s mean, ${(stats.bestEpisodeSpeed * 3.6).toFixed(1)} km/h best episode`,
    `stability ${pct(stats.fallRate)} of episodes ended in a fall, ${pct(stats.groundContact)} ground contact`,
    `score    ${stats.score.toFixed(3)}`,
  ].join('\n  ');
}
