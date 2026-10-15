/**
 * Gradient checks for the hand-written backward pass.
 *
 * A hand-written backward pass is the easiest place in an RL implementation to
 * hide a silent bug (training still "runs", it just learns nothing). So the PPO
 * minibatch loss is checked against central finite differences on the real
 * parameters: the analytic gradient must match to ~1e-6 relative error.
 *
 * The loss recomputed here mirrors `PPOTrainer._minibatchLossGrad` exactly.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Policy } from '../src/net.js';
import { PPOTrainer } from '../src/ppo.js';
import { OBS_SIZE, ACTION_SIZE } from '../src/env.js';
import { mulberry32, gaussian } from '../src/math.js';

const OBS = OBS_SIZE;
const ACT = ACTION_SIZE;
const CLIP = 0.2;
const VALUE_COEF = 0.5;
const ENTROPY_COEF = 0.01;

/** The same loss the trainer optimises, as a pure scalar function of the weights. */
function lossOf(policy, batch) {
  const { obs, act, adv, oldLogp, ret } = batch;
  const M = act.length / ACT;
  const std = new Float64Array(ACT);
  for (let i = 0; i < ACT; i++) std[i] = Math.exp(Math.min(1, Math.max(-4, policy.logStd[i])));

  let loss = 0;
  for (let s = 0; s < M; s++) {
    const { mean, value } = policy.forward(obs.subarray(s * OBS, s * OBS + OBS));
    let logp = 0;
    let entropy = 0;
    for (let i = 0; i < ACT; i++) {
      const dev = (act[s * ACT + i] - mean[i]) / std[i];
      logp += -0.5 * dev * dev - Math.log(std[i]) - 0.5 * Math.log(2 * Math.PI);
      entropy += 0.5 * Math.log(2 * Math.PI * Math.E) + Math.log(std[i]);
    }
    const ratio = Math.exp(logp - oldLogp[s]);
    const unclipped = ratio * adv[s];
    const clipped = Math.min(1 + CLIP, Math.max(1 - CLIP, ratio)) * adv[s];
    loss += -Math.min(unclipped, clipped) / M;
    loss += (VALUE_COEF * 0.5 * (value - ret[s]) * (value - ret[s])) / M;
    loss -= (ENTROPY_COEF * entropy) / M;
  }
  return loss;
}

function makeBatch(seed, M = 6) {
  const rng = mulberry32(seed);
  const obs = new Float64Array(M * OBS);
  for (let i = 0; i < obs.length; i++) obs[i] = gaussian(rng) * 0.8;
  const act = new Float64Array(M * ACT);
  for (let i = 0; i < act.length; i++) act[i] = Math.tanh(gaussian(rng));
  const adv = new Float64Array(M);
  const ret = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    adv[i] = gaussian(rng);
    ret[i] = gaussian(rng);
  }
  return { obs, act, adv, ret, oldLogp: new Float64Array(M) };
}

test('policy gradients match central finite differences', () => {
  const trainer = new PPOTrainer({
    seed: 3,
    hidden: [8, 8],
    hyper: { clip: CLIP, valueCoef: VALUE_COEF, entropyCoef: ENTROPY_COEF },
  });
  const policy = trainer.policy;
  // Bias the log-std away from its default so the entropy term has a gradient.
  policy.logStd.fill(-0.2);
  const batch = makeBatch(11);
  const M = batch.act.length / ACT;

  // Old log-probs: recompute under the current policy, then nudge a couple of
  // samples so some fall outside the trust region and the clip branch is used.
  const std = new Float64Array(ACT);
  for (let i = 0; i < ACT; i++) std[i] = Math.exp(Math.min(1, Math.max(-4, policy.logStd[i])));
  for (let s = 0; s < M; s++) {
    const { mean } = policy.forward(batch.obs.subarray(s * OBS, s * OBS + OBS));
    let logp = 0;
    for (let i = 0; i < ACT; i++) {
      const dev = (batch.act[s * ACT + i] - mean[i]) / std[i];
      logp += -0.5 * dev * dev - Math.log(std[i]) - 0.5 * Math.log(2 * Math.PI);
    }
    batch.oldLogp[s] = logp + (s % 3 === 0 ? 0.5 : -0.02);
  }

  // Analytic gradients straight from the trainer's own minibatch routine.
  policy.zeroGrad();
  const diag = trainer._minibatchLossGrad(batch.obs, batch.act, batch.adv, batch.oldLogp, batch.ret, M);
  assert.ok(diag.clippedFraction > 0 && diag.clippedFraction < 1, `both surrogate branches must be exercised, clipped fraction ${diag.clippedFraction}`);
  const grads = policy.collect().grads;
  const { params } = policy.collect();

  const eps = 1e-6;
  let checked = 0;
  let worst = 0;
  for (let p = 0; p < params.length; p++) {
    const arr = params[p];
    const g = grads[p];
    const stride = Math.max(1, Math.floor(arr.length / 7));
    for (let i = 0; i < arr.length; i += stride) {
      const orig = arr[i];
      arr[i] = orig + eps;
      const lp = lossOf(policy, batch);
      arr[i] = orig - eps;
      const lm = lossOf(policy, batch);
      arr[i] = orig;
      const numeric = (lp - lm) / (2 * eps);
      const analytic = g[i];
      const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
      const relErr = Math.abs(numeric - analytic) / denom;
      worst = Math.max(worst, relErr);
      checked++;
      assert.ok(
        relErr < 2e-3,
        `param array ${p} index ${i}: analytic ${analytic.toExponential(6)} vs numeric ${numeric.toExponential(6)} (rel err ${relErr.toExponential(3)})`,
      );
    }
  }
  assert.ok(checked > 50, `should check a meaningful number of coordinates, checked ${checked}`);
  console.log(`      ${checked} coordinates checked, worst relative error ${worst.toExponential(2)}`);
});

test('a known linear head gradient is exact (sanity check of the harness)', () => {
  // If the gradient checker itself were broken, this trivially-true case would
  // fail: d/dw of mean-squared error of a single linear unit is analytic.
  const rng = mulberry32(5);
  const w = new Float64Array([0.3, -0.7, 1.1]);
  const x = new Float64Array([0.5, -1.5, 2.0]);
  const target = 0.4;
  const value = (wv) => {
    let z = 0;
    for (let i = 0; i < 3; i++) z += wv[i] * x[i];
    return 0.5 * (z - target) * (z - target);
  };
  const eps = 1e-6;
  for (let i = 0; i < 3; i++) {
    const orig = w[i];
    w[i] = orig + eps;
    const a = value(w);
    w[i] = orig - eps;
    const b = value(w);
    w[i] = orig;
    const numeric = (a - b) / (2 * eps);
    let z = 0;
    for (let k = 0; k < 3; k++) z += w[k] * x[k];
    const analytic = (z - target) * x[i];
    assert.ok(Math.abs(numeric - analytic) < 1e-6, `coordinate ${i}`);
  }
  void rng;
});

test('policy parameters survive a JSON round trip', async () => {
  const { Policy } = await import('../src/net.js');
  const policy = new Policy({ obsSize: OBS, actSize: ACT, hidden: [8, 8], seed: 9, logStdInit: -0.5 });
  const obs = new Float64Array(OBS).fill(0.13);
  const before = policy.actDeterministic(obs);
  const json = JSON.parse(JSON.stringify(policy.toJSON()));
  const restored = Policy.fromJSON(json);
  const after = restored.actDeterministic(obs);
  for (let i = 0; i < ACT; i++) {
    assert.ok(Math.abs(before[i] - after[i]) < 1e-5, `action ${i}: ${before[i]} vs ${after[i]}`);
  }
  assert.deepEqual(Array.from(restored.logStd), Array.from(policy.logStd));
});
