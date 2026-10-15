/**
 * Integration test for the browser lane worker, executed in Node.
 *
 * `public/lane.js` is written for a Web Worker (`self.onmessage`,
 * `self.postMessage`). Stubbing those two globals lets the exact same file run
 * here, which is the cheapest possible way to catch a mismatch between the
 * worker and the trainer API - the kind of bug that would otherwise only show up
 * as a silent blank canvas in a browser.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

test('the browser lane worker trains a policy and reports status', async () => {
  const messages = [];
  let onMessage = null;
  globalThis.self = {
    postMessage: (msg) => messages.push(msg),
    close: () => {},
    set onmessage(fn) {
      onMessage = fn;
    },
    get onmessage() {
      return onMessage;
    },
  };

  // Cache-busting query: Node caches ES modules, and the worker registers its
  // handler as a module side effect, so each test needs a fresh evaluation.
  await import('../public/lane.js?suite=1');
  assert.equal(typeof onMessage, 'function', 'the worker must register a message handler');

  const send = (data) => onMessage({ data });

  send({ type: 'init', laneId: 0, name: 'test-lane', color: '#4c9bff', seed: 5, speed: 1, turbo: true, maxSteps: 1_500_000 });
  send({ type: 'start', maxSteps: 1_500_000 });

  // Let it chew through a few slices of real work.
  const deadline = Date.now() + 4000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
    const latest = messages.filter((m) => m.type === 'status').pop();
    if (latest && latest.steps > 3000) break;
  }

  const ready = messages.find((m) => m.type === 'ready');
  assert.ok(ready, 'worker must announce itself');
  const statuses = messages.filter((m) => m.type === 'status');
  assert.ok(statuses.length > 2, `worker must report status repeatedly, got ${statuses.length}`);
  const last = statuses[statuses.length - 1];

  assert.equal(last.laneId, 0);
  assert.ok(last.steps > 0, `steps should advance, got ${last.steps}`);
  assert.ok(last.updates > 0, `the PPO update must have run, got ${last.updates}`);
  assert.ok(Number.isFinite(last.kl), 'KL must be a number');
  assert.ok(last.params > 1000, `policy parameter count looks wrong: ${last.params}`);
  assert.ok(Array.isArray(last.history), 'history must be an array for the chart');

  // The showcase snapshot is what the canvas draws: check its shape and that it
  // is JSON-safe (a Map or a typed array would silently render nothing).
  const snap = last.showcase.snapshot;
  assert.ok(snap && Array.isArray(snap.bodies), 'snapshot.bodies must be a plain array');
  assert.equal(snap.bodies.length, 7, 'seven robot bodies');
  assert.ok(Array.isArray(snap.contacts) && snap.contacts.length === 2, 'two foot contact flags');
  assert.ok(JSON.stringify(snap).length > 100, 'snapshot must serialise');
  for (const body of snap.bodies) {
    assert.ok(Array.isArray(body.s) && body.s.length > 0, `body ${body.n} needs shapes`);
    for (const shape of body.s) {
      assert.equal(shape.length, 5, 'each shape is [ax, ay, bx, by, r]');
      for (const v of shape) assert.ok(Number.isFinite(v), `shape value must be finite in ${body.n}`);
    }
    for (const key of ['x', 'y', 'a']) assert.ok(Number.isFinite(body[key]), `${body.n}.${key} must be finite`);
  }

  // The visible rollout must actually be moving: after a few seconds of turbo
  // training, the showcase pace should be finite and non-negative.
  assert.ok(Number.isFinite(last.showcase.pace) && last.showcase.pace >= 0, `pace ${last.showcase.pace}`);

  send({ type: 'pause' });
  send({ type: 'checkpoint' });
  await new Promise((r) => setTimeout(r, 100));
  const checkpoint = messages.find((m) => m.type === 'checkpoint');
  assert.ok(checkpoint, 'checkpoint must be produced on request');
  assert.equal(checkpoint.policy.format, 'rl-running-policy@1');
  assert.equal(checkpoint.policy.obsSize, 23);
  assert.equal(checkpoint.policy.actSize, 6);
  assert.equal(checkpoint.policy.trunk.length, 2, 'two hidden layers');
  assert.ok(checkpoint.policy.logStd.length === 6, 'one log-std per action');

  send({ type: 'stop' });
  delete globalThis.self;
});

test('a checkpoint can be loaded back into the worker', async () => {
  const messages = [];
  let onMessage = null;
  globalThis.self = {
    postMessage: (msg) => messages.push(msg),
    close: () => {},
    set onmessage(fn) {
      onMessage = fn;
    },
    get onmessage() {
      return onMessage;
    },
  };
  await import('../public/lane.js?suite=2');
  const send = (data) => onMessage({ data });

  send({ type: 'init', laneId: 1, seed: 3, speed: 1, maxSteps: 100000 });
  await new Promise((r) => setTimeout(r, 250));
  send({ type: 'checkpoint' });
  await new Promise((r) => setTimeout(r, 120));
  const checkpoint = messages.find((m) => m.type === 'checkpoint');
  assert.ok(checkpoint, 'need a checkpoint to reload');

  // Re-init from that policy: this exercises Policy.fromJSON inside the worker.
  messages.length = 0;
  send({ type: 'init', laneId: 1, seed: 3, speed: 1, policyJSON: checkpoint.policy, maxSteps: 100000 });
  send({ type: 'start', maxSteps: 100000 });
  const deadline = Date.now() + 3000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
    const latest = messages.filter((m) => m.type === 'status').pop();
    if (latest && latest.steps > 1000) break;
  }
  const last = messages.filter((m) => m.type === 'status').pop();
  assert.ok(last && last.steps > 0, 'training must resume after loading a checkpoint');
  send({ type: 'stop' });
  delete globalThis.self;
});
