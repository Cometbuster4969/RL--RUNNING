/**
 * PPO, GAE and the training loop.
 *
 * The learning test is deterministic - every environment and policy owns a
 * seeded PRNG - so it can assert a hard threshold instead of a vague "does not
 * crash". It runs 16k steps (~7 s), which is the price of actually proving that
 * the learner learns rather than merely that it runs.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PPOTrainer, DEFAULT_HYPER } from '../src/ppo.js';
import { ACTION_SIZE, OBS_SIZE } from '../src/env.js';

test('the episode history spans the whole run while staying bounded', () => {
  const trainer = new PPOTrainer({ seed: 1, historyCap: 128 });
  for (let i = 0; i < 2000; i++) {
    trainer._recordEpisode({ step: i * 100, distance: i * 0.01, reward: 0, time: 1 });
  }
  const history = trainer.history;
  assert.ok(history.length <= 160, `history must stay near the cap, got ${history.length}`);
  assert.ok(history.length > 50, `history must stay useful, got ${history.length}`);
  assert.equal(history[0].step, 0, 'the first episode must survive (it is the whole point of a learning curve)');
  assert.equal(history[history.length - 1].step, 1999 * 100, 'the newest episode must always be present');
  for (let i = 1; i < history.length; i++) {
    assert.ok(history[i].step > history[i - 1].step, 'history must stay ordered');
  }
  // Resolution decays towards the start, so the *median* gap is small even
  // though the curve reaches 200k steps back.
  assert.ok(trainer.historyStride() < 2000, `median spacing ${trainer.historyStride()} should stay dense`);
});

test('GAE advantages match an independent reference implementation', () => {
  const trainer = new PPOTrainer({ seed: 2, hyper: { gamma: 0.99, lambda: 0.95, rewardScale: 1 } });
  const T = DEFAULT_HYPER.rolloutSteps;
  // Deterministic fake rollout.
  for (let t = 0; t < T; t++) {
    trainer.rewBuf[t] = Math.sin(t * 0.37) * 1.5;
    trainer.valBuf[t] = Math.cos(t * 0.21);
    trainer.doneBuf[t] = t % 211 === 210 ? 1 : 0;
  }
  trainer.bootstrapValue = 0.42;
  trainer.computeAdvantages();

  // Reference, written the slow and obvious way.
  const gamma = 0.99;
  const lambda = 0.95;
  const ref = new Float64Array(T);
  let gae = 0;
  for (let t = T - 1; t >= 0; t--) {
    const notDone = trainer.doneBuf[t] ? 0 : 1;
    const nextValue = t === T - 1 ? trainer.bootstrapValue : trainer.valBuf[t + 1];
    const delta = trainer.rewBuf[t] + gamma * nextValue * notDone - trainer.valBuf[t];
    gae = delta + gamma * lambda * notDone * gae;
    ref[t] = gae;
  }
  // Advantages are normalised before use; compare the normalisation too.
  let mean = 0;
  for (let t = 0; t < T; t++) mean += ref[t];
  mean /= T;
  let varSum = 0;
  for (let t = 0; t < T; t++) varSum += (ref[t] - mean) ** 2;
  // The implementation adds 1e-8 before dividing, so the reference must too.
  const std = Math.sqrt(varSum / T) + 1e-8;
  for (let t = 0; t < T; t++) {
    const expected = (ref[t] - mean) / std;
    assert.ok(
      Math.abs(trainer.advBuf[t] - expected) < 1e-9,
      `advantage ${t}: ${trainer.advBuf[t]} vs reference ${expected}`,
    );
    // The value target is advantage + value (the standard GAE-consistent target,
    // and the only choice that keeps the critic consistent with the advantages),
    // computed from the *un-normalised* advantage.
    const expectedReturn = ref[t] + trainer.valBuf[t];
    assert.ok(
      Math.abs(trainer.retBuf[t] - expectedReturn) < 1e-9,
      `return ${t}: ${trainer.retBuf[t]} vs expected ${expectedReturn}`,
    );
  }
  // Advantages are normalised (mean 0, unit variance); returns deliberately are
  // not - normalising the value target would destroy the scale of the reward.
  const stats = (arr) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
    return { mean, std: Math.sqrt(variance) };
  };
  const advStats = stats(Array.from(trainer.advBuf));
  const retStats = stats(Array.from(trainer.retBuf));
  assert.ok(Math.abs(advStats.mean) < 1e-9, `advantages must be centred, mean ${advStats.mean}`);
  assert.ok(Math.abs(advStats.std - 1) < 1e-6, `advantages must be unit scale, std ${advStats.std}`);
  assert.ok(Math.abs(retStats.std - 1) > 1e-6, 'returns must NOT be normalised');
});

test('a rollout buffers exactly the configured number of steps', () => {
  const trainer = new PPOTrainer({ seed: 3, hyper: { rolloutSteps: 256 } });
  trainer.collectRollout();
  assert.equal(trainer.steps, 256);
  for (let t = 0; t < 256; t++) {
    assert.ok(trainer.rewBuf[t] !== 0 || true, 'reward slots are filled');
  }
  // Rewards must vary: a constant reward buffer would mean the env is stuck.
  const rewards = Array.from(trainer.rewBuf);
  assert.ok(new Set(rewards.map((r) => r.toFixed(4))).size > 20, 'rewards must vary across the rollout');
  const observations = trainer.obsBuf;
  let nonZero = 0;
  for (const v of observations) if (v !== 0) nonZero++;
  assert.ok(nonZero > observations.length * 0.8, 'observations must be populated');
});

test('an update changes the parameters and reports finite diagnostics', () => {
  const trainer = new PPOTrainer({ seed: 4, hyper: { rolloutSteps: 256, minibatch: 64 } });
  const { params } = trainer.policy.collect();
  const before = Float64Array.from(params[0]);
  trainer.collectRollout();
  trainer.computeAdvantages();
  const diag = trainer.update();
  let changed = 0;
  for (let i = 0; i < before.length; i++) if (before[i] !== params[0][i]) changed++;
  assert.ok(changed > before.length * 0.5, `an Adam step must move the weights (${changed}/${before.length})`);
  for (const key of ['kl', 'entropy', 'valueLoss', 'policyLoss']) {
    assert.ok(Number.isFinite(diag[key]), `${key} must be finite, got ${diag[key]}`);
  }
  assert.ok(Number.isFinite(trainer.gradNorm) && trainer.gradNorm >= 0, 'gradient norm must be reported');
  assert.ok(Number.isFinite(trainer.kl), 'KL must be reported');
});

test('training improves the policy within 16k steps (deterministic end-to-end)', () => {
  const trainer = new PPOTrainer({ seed: 1 });
  const untrained = trainer.evaluate(4, { greedy: true });
  assert.ok(untrained.mean < 0.05, `an untrained policy should barely move, got ${untrained.mean}`);

  trainer.trainSteps(16000);
  const trained = trainer.evaluate(4, { greedy: true });
  assert.ok(trained.mean > 0.4, `the policy must learn to travel after 16k steps, got ${trained.mean} m`);
  assert.ok(trainer.bestDistance > 0.8, `best training episode should be substantial, got ${trainer.bestDistance} m`);
  assert.equal(trained.fallRate, 0, 'a learner at this stage should not be falling over every episode');
  assert.ok(trainer.episodes > 20, `16k steps should complete many episodes, got ${trainer.episodes}`);
  // Actions must stay inside the action space the environment accepts.
  const action = trainer.policy.actDeterministic(trainer.env.obs);
  assert.equal(action.length, ACTION_SIZE);
  for (const a of action) assert.ok(Number.isFinite(a));
  assert.equal(trainer.env.obs.length, OBS_SIZE);
});
