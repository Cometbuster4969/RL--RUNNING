/**
 * Main thread: owns the lanes, draws the track, keeps the leaderboard and the
 * learning curve. All the learning happens in the lane workers; this file only
 * renders what they report.
 */
// The dev server serves src/ alongside public/, so a relative import here means
// the browser uses the very same helper the Node scripts use - and Node can
// still resolve this file, which is what makes the UI testable.
import { formatCount } from '../src/math.js';

const LANE_COLORS = ['#4c9bff', '#43d17c', '#f0b429', '#c77dff'];
const TRACK_METRES = 13; // world metres visible across a lane canvas
const WORLD_HEIGHT = 2.1;
const LAP_METRES = 25;

const ui = {
  start: document.getElementById('btn-start'),
  reset: document.getElementById('btn-reset'),
  champion: document.getElementById('btn-champion'),
  laneCount: document.getElementById('lane-count'),
  mode: document.getElementById('mode'),
  speedButtons: [...document.querySelectorAll('.segmented button')],
  track: document.getElementById('track'),
  boardBody: document.querySelector('#board tbody'),
  chart: document.getElementById('chart'),
  log: document.getElementById('log'),
  footerStats: document.getElementById('footer-stats'),
  perfNote: document.getElementById('perf-note'),
};

/** @type {Map<number, any>} */
const lanes = new Map();
let running = false;
let speed = 2;
let turbo = true;
let startedAt = null;
let totalSteps = 0;
let bestPaceSeen = 0;
let lastLogAt = 0;

/* ------------------------------------------------------------------ lanes */

function laneTemplate(index, color) {
  const el = document.createElement('section');
  el.className = 'lane';
  el.dataset.lane = String(index);
  el.innerHTML = `
    <div class="lane-head">
      <span class="lane-title"><i class="swatch" style="background:${color}"></i>Agent ${index + 1}</span>
      <span class="badge flight">FLIGHT PHASE</span>
      <span class="stat">pace <b data-field="pace">0.00</b> m/s</span>
      <span class="stat">speed <b data-field="kmh">0.0</b> km/h</span>
      <span class="stat">run <b data-field="distance">0.00</b> m</span>
      <span class="stat">falls <b data-field="falls">0</b></span>
      <span class="spacer"></span>
      <span class="stat">episodes trained <b data-field="episodes">0</b></span>
      <span class="stat">steps <b data-field="steps">0</b></span>
      <span class="stat">best <b data-field="best">0.00</b> m</span>
    </div>
    <canvas></canvas>
  `;
  return el;
}

function createLane(index, opts = {}) {
  destroyLane(index);
  const color = LANE_COLORS[index % LANE_COLORS.length];
  const el = laneTemplate(index, color);
  ui.track.appendChild(el);
  const canvas = el.querySelector('canvas');
  const worker = new Worker('/lane.js', { type: 'module' });

  const lane = {
    index,
    color,
    el,
    canvas,
    ctx: canvas.getContext('2d'),
    worker,
    status: null,
    prevSnapshot: null,
    snapshot: null,
    snapshotAt: 0,
    prevAt: 0,
    flightUntil: 0,
    laneDistance: 0,
    lastEpisode: -1,
  };
  lanes.set(index, lane);

  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'status') onStatus(lane, msg);
    else if (msg.type === 'checkpoint') onCheckpoint(lane, msg);
  };
  worker.onerror = (err) => log(`Agent ${index + 1} worker error: ${err.message}`);

  worker.postMessage({
    type: 'init',
    laneId: index,
    name: `agent-${index + 1}`,
    color,
    seed: opts.seed ?? 1 + index * 977,
    speed,
    turbo,
    maxSteps: opts.maxSteps ?? Infinity,
    policyJSON: opts.policyJSON ?? null,
    policyPerturb: opts.policyPerturb ?? 0,
  });
}

function destroyLane(index) {
  const lane = lanes.get(index);
  if (!lane) return;
  lane.worker.postMessage({ type: 'stop' });
  lane.worker.terminate();
  lane.el.remove();
  lanes.delete(index);
}

function onStatus(lane, msg) {
  lane.status = msg;
  lane.prevSnapshot = lane.snapshot;
  lane.prevAt = lane.snapshotAt;
  lane.snapshot = msg.showcase.snapshot;
  lane.snapshotAt = performance.now();
  lane.laneDistance = Math.max(lane.laneDistance, msg.showcase.distance);

  const contacts = msg.showcase.snapshot.contacts;
  if (contacts[0] === 0 && contacts[1] === 0 && msg.showcase.pace > 0.8) lane.flightUntil = performance.now() + 300;

  // Log notable events: falls, personal bests, ramp-ups.
  const now = performance.now();
  const prevEpisodes = lane.lastEpisodes ?? 0;
  if (msg.episodes > prevEpisodes) lane.lastEpisodes = msg.episodes;
  if (msg.showcase.falls > (lane.lastFalls ?? 0)) {
    lane.lastFalls = msg.showcase.falls;
    if (now - lastLogAt > 4000) {
      log(`Agent ${lane.index + 1} went down after ${msg.showcase.distance.toFixed(2)} m (episode ${msg.showcase.episode}).`);
      lastLogAt = now;
    }
  }
  if (msg.showcase.pace > bestPaceSeen + 0.35 && msg.showcase.pace > 1.2) {
    bestPaceSeen = msg.showcase.pace;
    log(
      `Agent ${lane.index + 1} is running at ${msg.showcase.pace.toFixed(2)} m/s (${msg.showcase.kmh.toFixed(1)} km/h) after ${formatCount(msg.steps)} training steps.`,
    );
    lastLogAt = now;
  }
  if (msg.best > (lane.bestLogged ?? 0) + 5) {
    lane.bestLogged = msg.best;
    log(`Agent ${lane.index + 1} completed a ${msg.best.toFixed(1)} m training episode (${formatCount(msg.steps)} steps).`);
  }

  updateHeader(lane, msg);
}

function onCheckpoint(lane, msg) {
  const blob = new Blob([JSON.stringify(msg.policy)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `agent-${lane.index + 1}-${formatCount(msg.steps)}steps.json`;
  a.click();
  URL.revokeObjectURL(url);
  log(`Saved agent ${lane.index + 1}: ${formatCount(msg.steps)} steps, best episode ${msg.best.toFixed(2)} m.`);
}

function updateHeader(lane, msg) {
  const set = (field, value) => {
    const el = lane.el.querySelector(`[data-field="${field}"]`);
    if (el) el.textContent = value;
  };
  set('pace', msg.showcase.pace.toFixed(2));
  set('kmh', msg.showcase.kmh.toFixed(1));
  set('distance', msg.showcase.distance.toFixed(2));
  set('falls', String(msg.showcase.falls));
  set('episodes', formatCount(msg.episodes));
  set('steps', formatCount(msg.steps));
  set('best', msg.best.toFixed(2));
  const badge = lane.el.querySelector('.badge.flight');
  badge.classList.toggle('on', performance.now() < lane.flightUntil);
}

/* --------------------------------------------------------------- rendering */

function fitCanvas(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(320, Math.round(rect.width * dpr));
  const h = Math.max(120, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { w, h, dpr };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

/**
 * Draw one lane. The camera follows the showcase robot and snaps to a one metre
 * grid, which makes the world scroll smoothly and gives an honest sense of speed.
 */
function drawLane(lane, now) {
  const { ctx, canvas } = lane;
  const { w, h } = fitCanvas(canvas);
  const snapshot = lane.snapshot;
  ctx.clearRect(0, 0, w, h);

  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, '#101728');
  gradient.addColorStop(1, '#0b0e15');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  if (!snapshot) return;

  const scale = h / WORLD_HEIGHT;
  // Interpolate between the last two snapshots for smooth motion at ~10 Hz.
  // A reset moves the robot back to the start line in one report: snapping
  // instead of interpolating avoids a fast backwards camera sweep.
  const resetted = lane.prevSnapshot && lane.snapshot && Math.abs(lane.snapshot.x - lane.prevSnapshot.x) > 2;
  const dt = Math.max(1, lane.snapshotAt - lane.prevAt);
  const t = lane.prevSnapshot && !resetted ? Math.min(1, (now - lane.snapshotAt) / dt) : 1;
  const anchor = lane.prevSnapshot && !resetted ? lerp(lane.prevSnapshot.x, snapshot.x, t) : snapshot.x;
  const groundY = Math.round(h * 0.86);
  const cameraX = Math.floor(anchor) - TRACK_METRES * 0.35;
  const toScreenX = (x) => (x - cameraX) * scale;
  const toScreenY = (y) => groundY - y * scale;

  // Ground + metre markers.
  ctx.fillStyle = '#222c40';
  ctx.fillRect(0, groundY, w, 2);
  ctx.font = `${Math.round(11 * (w / 900))}px ui-monospace, monospace`;
  for (let m = Math.floor(cameraX); m < cameraX + TRACK_METRES + 1; m++) {
    const x = toScreenX(m);
    const major = m % 5 === 0;
    ctx.fillStyle = major ? 'rgba(150,170,205,0.55)' : 'rgba(90,105,135,0.35)';
    ctx.fillRect(x, groundY, 1, major ? 12 : 7);
    if (major) {
      ctx.fillStyle = 'rgba(150,170,205,0.5)';
      ctx.fillText(`${m} m`, x + 4, groundY + 26);
    }
  }

  // The robot: interpolated body poses, drawn from its real capsule shapes.
  const bodies = [];
  for (let i = 0; i < snapshot.bodies.length; i++) {
    const body = snapshot.bodies[i];
    const prev = lane.prevSnapshot && !resetted ? lane.prevSnapshot.bodies[i] : null;
    bodies.push({
      ...body,
      x: prev ? lerp(prev.x, body.x, t) : body.x,
      y: prev ? lerp(prev.y, body.y, t) : body.y,
      a: prev ? lerpAngle(prev.a, body.a, t) : body.a,
    });
  }
  const byName = new Map(bodies.map((b) => [b.n, b]));
  ctx.lineCap = 'round';
  for (const body of bodies) {
    const isFarLeg = body.g === 'far';
    ctx.strokeStyle = bodyColor(body, lane.color);
    ctx.globalAlpha = isFarLeg ? 0.65 : 1;
    for (const [ax, ay, bx, by, r] of body.s) {
      const c = Math.cos(body.a);
      const s = Math.sin(body.a);
      // Local coordinates are relative to the body centre of mass.
      const wx0 = body.x + c * ax - s * ay;
      const wy0 = body.y + s * ax + c * ay;
      const wx1 = body.x + c * bx - s * by;
      const wy1 = body.y + s * bx + c * by;
      ctx.lineWidth = Math.max(2, r * 2 * scale);
      ctx.beginPath();
      ctx.moveTo(toScreenX(wx0), toScreenY(wy0));
      ctx.lineTo(toScreenX(wx1), toScreenY(wy1));
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  // Contact sparks and a flight indicator.
  const contacts = snapshot.contacts;
  for (let l = 0; l < contacts.length; l++) {
    if (!contacts[l]) continue;
    const foot = byName.get(l === 0 ? 'foot_far' : 'foot_near');
    if (!foot) continue;
    const x = toScreenX(foot.x);
    const y = toScreenY(foot.y);
    ctx.fillStyle = l === 0 ? 'rgba(190,150,70,0.75)' : 'rgba(255,205,90,0.95)';
    ctx.beginPath();
    ctx.arc(x, y + 4, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Progress bar towards the current lap.
  const lapProgress = (lane.laneDistance % LAP_METRES) / LAP_METRES;
  ctx.fillStyle = 'rgba(255,255,255,0.07)';
  ctx.fillRect(0, 0, w, 3);
  ctx.fillStyle = lane.color;
  ctx.fillRect(0, 0, w * lapProgress, 3);
}

function bodyColor(body, laneColor) {
  if (body.n.startsWith('foot')) return '#f0b429';
  if (body.g === 'trunk') return body.n === 'trunk' ? laneColor : '#f0b429';
  if (body.g === 'far') return '#2f6ba8';
  return '#6fb0f0';
}

/** Animation loop: redraw every lane every frame. */
function frame() {
  const now = performance.now();
  for (const lane of lanes.values()) {
    drawLane(lane, now);
    const badge = lane.el.querySelector('.badge.flight');
    badge.classList.toggle('on', now < lane.flightUntil);
  }
  drawChart();
  requestAnimationFrame(frame);
}

/* --------------------------------------------------------------- the board */

function renderBoard() {
  const rows = [...lanes.values()]
    .map((lane) => ({
      lane,
      pace: lane.status?.showcase.pace ?? 0,
      kmh: lane.status?.showcase.kmh ?? 0,
      best: lane.status?.best ?? 0,
      steps: lane.status?.steps ?? 0,
    }))
    .sort((a, b) => b.pace - a.pace);

  ui.boardBody.innerHTML = rows
    .map((row, i) => {
      const medal = i === 0 ? ' class="medal"' : '';
      return `<tr class="${i === 0 && rows.length > 1 ? 'leader' : ''}">
        <td${medal}>${i + 1}</td>
        <td class="name"><span class="swatch" style="display:inline-block;background:${row.lane.color}"></span> Agent ${row.lane.index + 1}</td>
        <td class="num">${row.pace.toFixed(2)} m/s</td>
        <td class="num">${row.kmh.toFixed(1)}</td>
        <td class="num">${row.best.toFixed(1)} m</td>
        <td class="num">${formatCount(row.steps)}</td>
      </tr>`;
    })
    .join('');
}

/* ------------------------------------------------------------- the chart */

let chartData = null;
function drawChart() {
  const canvas = ui.chart;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.round(rect.width * dpr);
  const h = Math.round(360 * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const padL = 46 * dpr;
  const padR = 14 * dpr;
  const padT = 14 * dpr;
  const padB = 30 * dpr;

  let maxStep = 10000;
  let maxDist = 5;
  for (const lane of lanes.values()) {
    const history = lane.status?.history;
    if (!history || !history.length) continue;
    maxStep = Math.max(maxStep, history[history.length - 1][0]);
    for (const [, dist] of history) maxDist = Math.max(maxDist, dist);
  }
  maxDist = Math.ceil(maxDist * 1.15);

  const X = (step) => padL + (step / maxStep) * (w - padL - padR);
  const Y = (dist) => h - padB - (dist / maxDist) * (h - padT - padB);

  // grid
  ctx.strokeStyle = 'rgba(36,49,74,0.8)';
  ctx.fillStyle = 'rgba(150,170,205,0.65)';
  ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.lineWidth = 1;
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const dist = (maxDist / yTicks) * i;
    const y = Y(dist);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(`${dist.toFixed(0)} m`, 6 * dpr, y + 4 * dpr);
  }
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const step = (maxStep / xTicks) * i;
    const x = X(step);
    ctx.fillText(formatCount(step), x - 12 * dpr, h - 10 * dpr);
  }
  ctx.fillText('training steps', w / 2 - 30 * dpr, h - 2 * dpr);

  // per-lane curves
  for (const lane of lanes.values()) {
    const history = lane.status?.history;
    if (!history || history.length < 2) continue;
    ctx.strokeStyle = lane.color;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    history.forEach(([step, dist], i) => {
      const x = X(step);
      const y = Y(dist);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // running best
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    let best = 0;
    history.forEach(([step, dist], i) => {
      best = Math.max(best, dist);
      const x = X(step);
      const y = Y(best);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  chartData = { maxStep, maxDist };
}

/* ------------------------------------------------------------------- misc */

function log(message) {
  const li = document.createElement('li');
  const time = new Date().toLocaleTimeString();
  li.innerHTML = `<time>${time}</time>${message}`;
  ui.log.prepend(li);
  while (ui.log.children.length > 40) ui.log.lastChild.remove();
}

function updateFooter() {
  let steps = 0;
  let episodes = 0;
  for (const lane of lanes.values()) {
    steps += lane.status?.steps ?? 0;
    episodes += lane.status?.episodes ?? 0;
  }
  totalSteps = steps;
  const elapsed = startedAt ? (Date.now() - startedAt) / 1000 : 0;
  ui.footerStats.textContent = `${formatCount(steps)} training steps · ${formatCount(episodes)} episodes${
    elapsed > 2 ? ` · ${formatCount(steps / elapsed)} steps/s · ${formatDuration(elapsed)} elapsed` : ''
  }`;
}

/* --------------------------------------------------------------- controls */

function setRunning(next) {
  running = next;
  ui.start.textContent = running ? 'Pause' : 'Start';
  ui.start.classList.toggle('paused', !running);
  if (running && !startedAt) startedAt = Date.now();
  for (const lane of lanes.values()) lane.worker.postMessage({ type: running ? 'start' : 'pause' });
  if (running) log('Training started. Every agent starts from random weights and no idea how legs work.');
}

function resetAll() {
  log('Resetting all agents - new random weights, same rules.');
  startedAt = Date.now();
  bestPaceSeen = 0;
  ui.log.innerHTML = '';
  const count = Number(ui.laneCount.value);
  for (let i = 0; i < count; i++) {
    const lane = lanes.get(i);
    if (lane) {
      lane.laneDistance = 0;
      lane.status = null;
      lane.worker.postMessage({ type: 'reset', seed: 1 + i * 977 + Math.floor(Math.random() * 5000), andStart: running });
    }
  }
}

function rebuildLanes(count) {
  const wasRunning = running;
  for (let i = 0; i < 4; i++) if (i >= count) destroyLane(i);
  for (let i = 0; i < count; i++) if (!lanes.has(i)) createLane(i);
  startedAt = Date.now();
  if (wasRunning) setRunning(true);
}

/**
 * Warm start every lane from the reference policy in runs/, if the server has
 * one. This is the "show me a finished runner" button: cold start is the honest
 * default, but it is nice to be able to compare against a policy that has
 * already trained for millions of steps.
 */
async function loadChampion() {
  ui.champion.disabled = true;
  try {
    const res = await fetch('/api/champion?dir=champion&name=champion.json');
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const policy = await res.json();
    if (!policy || !policy.trunk) throw new Error('malformed checkpoint');
    let seed = 1000;
    for (const lane of lanes.values()) {
      lane.laneDistance = 0;
      lane.status = null;
      lane.worker.postMessage({
        type: 'reset',
        seed: (seed += 1013),
        policyJSON: policy,
        // Small perturbations so the lanes are still distinct learners.
        policyPerturb: 0.02,
        andStart: running,
      });
    }
    log(
      `Loaded the reference champion (${policy.hidden.join('x')} trunk, ${policy.trunk.length + 2} weight matrices) into ${lanes.size} agent(s).`,
    );
    log('They now start from a policy that can already run, and keep improving from there.');
  } catch (err) {
    log(`No champion to load yet (${err.message}). Train one first: <code>node scripts/train.mjs --lanes 2 --steps 1500000 --out runs/champion</code>`);
    ui.champion.disabled = false;
    return;
  }
  ui.champion.disabled = false;
}

ui.start.addEventListener('click', () => setRunning(!running));
ui.champion.addEventListener('click', loadChampion);
ui.reset.addEventListener('click', resetAll);
ui.laneCount.addEventListener('change', () => rebuildLanes(Number(ui.laneCount.value)));
ui.mode.addEventListener('change', () => {
  turbo = ui.mode.value === 'turbo';
  for (const lane of lanes.values()) lane.worker.postMessage({ type: 'turbo', value: turbo });
  log(
    turbo
      ? 'Turbo training: the learner runs flat out, the track replays the latest policy.'
      : 'Watch only: simulation advances at the speed of the animation, so you see it learn live.',
  );
});
for (const button of ui.speedButtons) {
  button.addEventListener('click', () => {
    speed = Number(button.dataset.speed);
    for (const b of ui.speedButtons) b.classList.toggle('active', b === button);
    for (const lane of lanes.values()) lane.worker.postMessage({ type: 'speed', value: speed });
  });
}

setInterval(updateFooter, 500);
setInterval(renderBoard, 500);

/* --------------------------------------------------------------- start up */

const params = new URLSearchParams(location.search);
if (params.get('lanes')) ui.laneCount.value = params.get('lanes');
if (params.get('speed')) {
  speed = Number(params.get('speed'));
  for (const b of ui.speedButtons) b.classList.toggle('active', Number(b.dataset.speed) === speed);
}
ui.mode.value = turbo ? 'turbo' : 'watch';

rebuildLanes(Number(ui.laneCount.value));
renderBoard();
requestAnimationFrame(frame);
updateFooter();

ui.perfNote.textContent =
  'Everything on this page - the physics, the network, the optimiser - is computed in your browser, in worker threads. Nothing is sent anywhere.';
log('Ready. Press Start to begin training.');
