/**
 * UI integration test.
 *
 * `public/app.js` is the only part of the project that cannot be exercised by
 * importing it in Node - it needs a document, a canvas and workers. This test
 * supplies a small stub of each, feeds the app real status messages from a real
 * lane, and asserts that the renderer actually draws: a typo in the canvas code
 * would otherwise show up only as a blank lane in a browser.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PPOTrainer } from '../src/ppo.js';

/** Minimal 2D context that records what was drawn. */
function makeContext() {
  const calls = [];
  const record = (name) => (...args) => {
    calls.push([name, ...args]);
  };
  return {
    calls,
    clearRect: record('clearRect'),
    fillRect: record('fillRect'),
    createLinearGradient: () => ({ addColorStop() {} }),
    beginPath: record('beginPath'),
    moveTo: record('moveTo'),
    lineTo: record('lineTo'),
    stroke: record('stroke'),
    fill: record('fill'),
    arc: record('arc'),
    fillText: record('fillText'),
    set fillStyle(v) {
      calls.push(['fillStyle', v]);
    },
    get fillStyle() {
      return '#000';
    },
    set strokeStyle(v) {
      calls.push(['strokeStyle', v]);
    },
    get strokeStyle() {
      return '#000';
    },
    lineWidth: 1,
    lineCap: 'butt',
    globalAlpha: 1,
    font: '',
  };
}

function makeElement(id = '') {
  const el = {
    id,
    textContent: '',
    innerHTML: '',
    disabled: false,
    dataset: {},
    style: {},
    children: [],
    value: '',
    className: '',
    classList: {
      add() {},
      remove() {},
      toggle() {},
      contains: () => false,
    },
    listeners: {},
    addEventListener(type, fn) {
      this.listeners[type] = fn;
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    remove() {},
    prepend(child) {
      this.children.unshift(child);
    },
    // Cache by selector: the app queries the same child repeatedly (canvas,
    // badges, stat fields) and must get the same element back each time.
    _childCache: null,
    querySelector(selector) {
      if (!this._childCache) this._childCache = new Map();
      if (!this._childCache.has(selector)) {
        const child = makeElement(`${id} ${selector}`);
        this._childCache.set(selector, child);
        this.children.push(child);
      }
      return this._childCache.get(selector);
    },
    querySelectorAll() {
      return [];
    },
    getBoundingClientRect: () => ({ width: 900, height: 172, top: 0, left: 0 }),
    getContext: () => el._ctx || (el._ctx = makeContext()),
    click() {},
    get lastChild() {
      return this.children[this.children.length - 1] ?? makeElement();
    },
  };
  return el;
}

/** A worker stub that answers init by emitting realistic status messages. */
class FakeWorker {
  constructor() {
    FakeWorker.instances.push(this);
    this.messages = [];
    this.onmessage = null;
    this.onerror = null;
  }
  postMessage(msg) {
    this.messages.push(msg);
    if (msg.type === 'init') {
      this.seed = msg.seed;
      this.laneId = msg.laneId;
      this.trainer = new PPOTrainer({ seed: msg.seed, randomize: true });
      // Report a couple of statuses straight away, like a real worker would.
      for (let i = 0; i < 3; i++) {
        this.trainer.trainSteps(256);
        queueMicrotask(() => this.onmessage && this.onmessage({ data: this.status() }));
      }
    }
  }
  status() {
    return {
      type: 'status',
      laneId: this.laneId,
      name: `agent-${this.laneId + 1}`,
      color: '#4c9bff',
      steps: this.trainer.steps,
      updates: this.trainer.updates,
      episodes: this.trainer.episodes,
      best: this.trainer.bestDistance,
      recent: this.trainer.distanceEma,
      kl: this.trainer.kl,
      entropy: this.trainer.entropy,
      valueLoss: this.trainer.valueLoss,
      params: this.trainer.policy.paramCount(),
      showcase: {
        x: this.trainer.env.trunk.x,
        xAbs: this.trainer.env.trunk.x,
        t: this.trainer.env.time,
        distance: 1.23,
        pace: 1.25,
        kmh: 4.5,
        episode: 3,
        falls: 1,
        episodeLog: [],
        snapshot: this.trainer.env.snapshot(),
      },
      learning: { episode: 1.1, time: 2.2, index: 3, completed: this.trainer.episodes },
      history: [[100, 0.5], [200, 1.5], [300, 2.25]],
      finished: false,
    };
  }
  terminate() {
    this.terminated = true;
  }
}
FakeWorker.instances = [];

test('the UI boots, receives lane statuses and draws to the canvas', async () => {
  const elements = new Map();
  const getElement = (id) => {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  };
  // The app looks these up by id and by selector.
  for (const id of ['btn-start', 'btn-reset', 'btn-champion', 'lane-count', 'mode', 'track', 'chart', 'log', 'footer-stats', 'perf-note']) {
    getElement(id);
  }
  getElement('lane-count').value = '2';

  const boardBody = makeElement('board-body');
  const speedButtons = [1, 2, 4, 8].map((s) => {
    const b = makeElement(`speed-${s}`);
    b.dataset.speed = String(s);
    return b;
  });

  let rafCallback = null;
  const frameCallbacks = [];
  globalThis.document = {
    getElementById: getElement,
    querySelector(selector) {
      if (selector === '#board tbody') return boardBody;
      return makeElement();
    },
    querySelectorAll(selector) {
      if (selector === '.segmented button') return speedButtons;
      return [];
    },
    createElement: () => makeElement(),
    body: makeElement(),
  };
  globalThis.window = { devicePixelRatio: 1, addEventListener() {} };
  globalThis.location = { search: '?lanes=2' };
  globalThis.Worker = FakeWorker;
  globalThis.requestAnimationFrame = (cb) => {
    rafCallback = cb;
    frameCallbacks.push(cb);
    return frameCallbacks.length;
  };
  globalThis.setInterval = () => 0;
  globalThis.clearInterval = () => {};
  globalThis.URL.createObjectURL = () => 'blob:stub';
  globalThis.URL.revokeObjectURL = () => {};
  globalThis.Blob = class {};

  await import('../public/app.js?suite=ui1');

  // A worker per lane, each initialised with a seed and the shared speed setting.
  assert.equal(FakeWorker.instances.length, 2, 'two lanes must be created');
  for (const worker of FakeWorker.instances) {
    const init = worker.messages.find((m) => m.type === 'init');
    assert.ok(init, 'each worker must be initialised');
    assert.equal(typeof init.seed, 'number');
    assert.ok(init.speed >= 1, 'animation speed must be passed to the worker');
  }

  // Let the queued status messages arrive.
  await new Promise((r) => setTimeout(r, 120));

  // Now drive the animation loop by hand: this is what paints the lanes.
  const track = getElement('track');
  assert.ok(track.children.length >= 2, `the track must contain a lane element per agent, got ${track.children.length}`);
  const laneEl = track.children[0];
  const canvas = laneEl.querySelector('canvas');
  assert.ok(canvas, 'each lane needs a canvas');

  assert.equal(typeof rafCallback, 'function', 'the app must register an animation frame');
  rafCallback();
  const ctx = canvas.getContext('2d');
  const names = ctx.calls.map((c) => c[0]);
  assert.ok(names.includes('fillRect'), 'the ground must be filled');
  assert.ok(names.includes('moveTo') && names.includes('lineTo') && names.includes('stroke'), 'the robot must be stroked');
  assert.ok(
    ctx.calls.filter((c) => c[0] === 'stroke').length > 6,
    `the robot has 7 bodies, each with at least one capsule: only ${ctx.calls.filter((c) => c[0] === 'stroke').length} strokes`,
  );
  // Body colours must come from the lane's palette.
  const strokeStyles = ctx.calls.filter((c) => c[0] === 'strokeStyle').map((c) => c[1]);
  assert.ok(strokeStyles.length >= 3, `expected several body colours, got ${strokeStyles.join(', ')}`);

  // The leaderboard and footer must reflect the reported numbers.
  assert.ok(boardBody.innerHTML.includes('Agent'))  , 'the leaderboard must list agents';
  assert.ok(getElement('footer-stats').textContent.includes('training steps'), 'the footer must show progress');
  assert.ok(getElement('log').children.length > 0, 'the activity log must have entries');

  // Starting training must reach the workers.
  const startButton = getElement('btn-start');
  assert.equal(typeof startButton.listeners.click, 'function', 'the Start button must be wired up');
  startButton.listeners.click();
  await new Promise((r) => setTimeout(r, 20));
  for (const worker of FakeWorker.instances) {
    assert.ok(worker.messages.some((m) => m.type === 'start'), 'pressing Start must tell every lane to start');
  }
  assert.equal(startButton.textContent, 'Pause', 'the button must flip to Pause');

  // Resetting must re-initialise, and pausing must reach every worker.
  startButton.listeners.click();
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(
    FakeWorker.instances.every((w) => w.messages.some((m) => m.type === 'pause')),
    'pressing Pause must reach every lane',
  );

  // Animation speed buttons must be forwarded.
  speedButtons[3].listeners.click();
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(
    FakeWorker.instances.some((w) => w.messages.some((m) => m.type === 'speed' && m.value === 8)),
    'speed changes must be forwarded to the lanes',
  );

  // Chart drawing must not throw, and must draw axes.
  assert.equal(typeof rafCallback, 'function');
  rafCallback();
  const chartCtx = getElement('chart').getContext('2d');
  assert.ok(chartCtx.calls.filter((c) => c[0] === 'fillText').length > 0, 'the chart must draw labels');

  delete globalThis.document;
  delete globalThis.Worker;
  delete globalThis.location;
});
