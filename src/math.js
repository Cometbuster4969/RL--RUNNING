/**
 * Small math helpers shared by the physics engine, the environment and the RL code.
 * No dependencies, plain ES modules - the same file runs in Node and in the browser.
 */

/** Clamp `v` into [lo, hi]. */
export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Linear interpolation. */
export function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Wrap an angle into (-PI, PI]. */
export function wrapAngle(a) {
  const x = (a + Math.PI) % (2 * Math.PI);
  return (x < 0 ? x + 2 * Math.PI : x) - Math.PI;
}

/** 2D cross product of a position vector and a vector: r x v. */
export function cross(rx, ry, vx, vy) {
  return rx * vy - ry * vx;
}

/**
 * Deterministic PRNG (mulberry32). Every training lane is seeded so runs are
 * reproducible: same seed + same code = same learning curve.
 * @param {number} seed
 * @returns {() => number} uniform sample in [0, 1)
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box-Muller (no cached spare, keeps the RNG stream simple). */
export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Integer in [0, n). */
export function randInt(rng, n) {
  return Math.floor(rng() * n) % n;
}

/** Human readable duration: 63 -> "1m03s". */
export function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

/** Compact number formatting for HUDs: 1234567 -> "1.23M". */
export function formatCount(n) {
  if (!isFinite(n)) return '-';
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return n.toFixed(0);
}
