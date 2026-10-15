/**
 * Physics regression tests. These encode the specific failure modes that were
 * hunted down while building the engine, so they cannot come back:
 *
 *  - resting bodies that spin up and launch themselves (angular velocity counted
 *    twice in the contact solver)
 *  - friction that collapses and kicks a sliding body when the normal impulse dips
 *  - Baumgarte push-out at the velocity level, which injects free energy
 *  - motors whose impulse accumulator saturates and then stops delivering torque
 *  - joint limits that creep past the limit instead of holding it
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { World, Body, RevoluteJoint } from '../src/physics.js';

const DT = 1 / 600;

function makeGround(world) {
  return world.addBody(
    new Body({ mass: 1, shapes: [{ a: [-100, -0.5], b: [100, -0.5], r: 0.5 }], y: 0, isStatic: true, name: 'ground' }),
  );
}

function makeBox(world, { y = 0.1, vx = 0, friction = 1.0 } = {}) {
  const box = new Body({
    mass: 1,
    shapes: [{ a: [-0.25, 0], b: [0.25, 0], r: 0.1 }],
    x: 0,
    y,
    friction,
    name: 'box',
  });
  world.addBody(box);
  box.vx = vx;
  return box;
}

test('a box dropped from 1 m lands and rests without bouncing or spinning', () => {
  const world = new World({ gravity: -9.81, iterations: 16 });
  makeGround(world);
  const box = makeBox(world, { y: 1.0 });
  let landed = -1;
  let peakAfterLanding = 0;
  for (let i = 0; i < 3000; i++) {
    world.step(DT);
    if (landed < 0 && box.y < 0.15) landed = i;
    if (landed >= 0 && i > landed + 30) peakAfterLanding = Math.max(peakAfterLanding, box.y);
  }
  assert.ok(landed > 0, 'box should reach the ground');
  assert.ok(Math.abs(box.vy) < 1e-3, `box should be at rest vertically, vy=${box.vy}`);
  assert.ok(Math.abs(box.omega) < 1e-3, `box should not spin, omega=${box.omega}`);
  assert.ok(peakAfterLanding < 0.105, `box should not bounce (peak ${peakAfterLanding})`);
  assert.ok(box.y > 0.096 && box.y < 0.101, `box should rest at its radius, y=${box.y}`);
});

test('a body placed exactly at rest stays exactly at rest', () => {
  const world = new World({ gravity: -9.81, iterations: 16 });
  makeGround(world);
  const box = makeBox(world, { y: 0.1 });
  for (let i = 0; i < 3000; i++) world.step(DT);
  assert.ok(Math.abs(box.omega) < 1e-6, `no rotation may appear, omega=${box.omega}`);
  assert.ok(Math.abs(box.vx) < 1e-6, `no sliding may appear, vx=${box.vx}`);
  assert.ok(box.y > 0.0999 && box.y < 0.1001, `height must not drift, y=${box.y}`);
});

test('sliding friction decelerates at mu*g', () => {
  const world = new World({ gravity: -9.81, iterations: 16 });
  makeGround(world);
  const box = makeBox(world, { y: 0.1, vx: 5 });
  let stopStep = -1;
  let maxY = 0;
  let minVx = Infinity;
  for (let i = 0; i < 1800; i++) {
    world.step(DT);
    maxY = Math.max(maxY, box.y);
    minVx = Math.min(minVx, box.vx);
    if (stopStep < 0 && box.vx < 1e-3) stopStep = i;
  }
  // v0 / (mu*g) = 5 / 9.81 = 0.51 s
  assert.ok(stopStep > 0, 'box must stop');
  assert.ok(Math.abs(stopStep * DT - 0.51) < 0.03, `stop time ${stopStep * DT}s should be ~0.51s`);
  assert.ok(Math.abs(box.vx) < 1e-3, `box must come to rest, vx=${box.vx}`);
  assert.ok(minVx > -1e-3, `friction must never reverse the box, min vx=${minVx}`);
  assert.ok(maxY < 0.102, `friction must not launch the box, peak y=${maxY}`);
});

test('frictionless contact preserves horizontal velocity', () => {
  const world = new World({ gravity: -9.81, iterations: 16 });
  makeGround(world);
  const box = makeBox(world, { y: 0.1, vx: 5, friction: 0 });
  for (let i = 0; i < 600; i++) world.step(DT);
  assert.ok(Math.abs(box.vx - 5) < 1e-3, `vx should be preserved, vx=${box.vx}`);
  assert.ok(box.y > 0.096 && box.y < 0.102, `height should be stable, y=${box.y}`);
});

test('a passive pendulum settles hanging straight down without gaining energy', () => {
  const world = new World({ gravity: -9.81, iterations: 20 });
  const anchor = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [0, 0], r: 0.05 }], y: 3, isStatic: true });
  const rod = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [1, 0], r: 0.05 }], x: 0, y: 3 });
  world.addBody(anchor);
  world.addBody(rod);
  const pin = world.addJoint(new RevoluteJoint(anchor, rod, rod.toWorldA(0, 0), { damping: 0.5 }));
  let maxAnchorError = 0;
  for (let i = 0; i < 9000; i++) {
    world.step(DT);
    const a = pin.bodyA.toWorld(pin.localAnchorA.x, pin.localAnchorA.y);
    const b = pin.bodyB.toWorld(pin.localAnchorB.x, pin.localAnchorB.y);
    maxAnchorError = Math.max(maxAnchorError, Math.hypot(a.x - b.x, a.y - b.y));
  }
  assert.ok(Math.abs(rod.angle + Math.PI / 2) < 0.01, `pendulum should hang down, angle=${rod.angle}`);
  assert.ok(maxAnchorError < 1e-3, `pin must not stretch, error=${maxAnchorError}`);
});

test('a motor holds its target angle indefinitely under gravity load', () => {
  const world = new World({ gravity: -9.81, iterations: 20 });
  const base = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [0, 0], r: 0.05 }], y: 2, isStatic: true });
  const limb = new Body({ mass: 4, shapes: [{ a: [0, 0], b: [0.5, 0], r: 0.05 }], x: 0, y: 2 });
  world.addBody(base);
  world.addBody(limb);
  const hip = world.addJoint(
    new RevoluteJoint(base, limb, limb.toWorldA(0, 0), { min: -1.5, max: 1.5, maxTorque: 60, target: 0.8 }),
  );
  for (let i = 0; i < 4500; i++) world.step(DT); // 7.5 simulated seconds
  assert.ok(Math.abs(hip.relativeAngle() - 0.8) < 0.01, `motor must hold target, angle=${hip.relativeAngle()}`);
  assert.ok(Math.abs(hip.relativeSpeed()) < 0.02, `motor must be settled, speed=${hip.relativeSpeed()}`);
});

test('a motor respects its torque cap', () => {
  const world = new World({ gravity: -9.81, iterations: 20 });
  const base = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [0, 0], r: 0.05 }], y: 2, isStatic: true });
  const limb = new Body({ mass: 20, shapes: [{ a: [0, 0], b: [1, 0], r: 0.05 }], x: 0, y: 2 });
  world.addBody(base);
  world.addBody(limb);
  const hip = world.addJoint(
    new RevoluteJoint(base, limb, limb.toWorldA(0, 0), { min: -1.5, max: 1.5, maxTorque: 5, target: 1.5 }),
  );
  for (let i = 0; i < 1800; i++) world.step(DT);
  assert.ok(hip.relativeAngle() < -1.0, `a weak motor must fail to lift a heavy limb, angle=${hip.relativeAngle()}`);
});

test('joint limits are held, not crept past', () => {
  const world = new World({ gravity: 0, iterations: 20 });
  const base = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [0, 0], r: 0.05 }], y: 2, isStatic: true });
  const limb = new Body({ mass: 1, shapes: [{ a: [0, 0], b: [0.5, 0], r: 0.05 }], x: 0, y: 2 });
  world.addBody(base);
  world.addBody(limb);
  const joint = world.addJoint(
    new RevoluteJoint(base, limb, limb.toWorldA(0, 0), { min: -0.5, max: 0.5, maxTorque: 500, target: 5 }),
  );
  // The hard stop holds the joint at `max` plus a small designed slop.
  for (let i = 0; i < 3000; i++) world.step(DT);
  assert.ok(
    joint.relativeAngle() > 0.499 && joint.relativeAngle() <= 0.506,
    `limit must hold at max, angle=${joint.relativeAngle()}`,
  );
  joint.target = -5;
  for (let i = 0; i < 3000; i++) world.step(DT);
  assert.ok(
    joint.relativeAngle() < -0.499 && joint.relativeAngle() >= -0.506,
    `lower limit must hold at min, angle=${joint.relativeAngle()}`,
  );
});

test('simulation is deterministic', () => {
  const run = () => {
    const world = new World({ gravity: -9.81, iterations: 16 });
    makeGround(world);
    const box = makeBox(world, { y: 1.3 });
    box.omega = 0.7;
    for (let i = 0; i < 2000; i++) world.step(DT);
    return [box.x, box.y, box.angle, box.vx, box.omega];
  };
  assert.deepEqual(run(), run());
});
