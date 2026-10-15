/**
 * The task: a planar (2D) biped has to learn to run to the right, as fast as it
 * can, without falling over.
 *
 * Morphology (7 rigid bodies, 6 actuated joints):
 *
 *          head
 *           |                      trunk (root body, one capsule + one head capsule)
 *           |                      hip x2, knee x2, ankle x2
 *        [trunk]
 *        /     \\   <- hips
 *     thigh   thigh
 *       |       |
 *     shin    shin
 *       |       |
 *     foot    foot
 *
 * Observations (23): trunk orientation/velocity/height, both legs' joint angles
 * and speeds, foot contact flags, and a phase clock (which makes periodic gaits
 * much easier to discover).
 *
 * Actions (6): target angles for the six joints. Every joint is a torque limited
 * motor, so the same action means the same thing at any control rate.
 *
 * Reward (per second): forward velocity, minus penalties for leaning over,
 * squatting, and spending effort, plus a bonus for staying upright. Falling ends
 * the episode with a fixed penalty, so "run fast" only pays if it is sustainable.
 */
import { World, Body, RevoluteJoint } from './physics.js';
import { clamp, mulberry32 } from './math.js';

export const CONTROL_DT = 1 / 60;
export const SUBSTEPS = 5;
export const PHYSICS_DT = CONTROL_DT / SUBSTEPS;
export const CLOCK_PERIOD = 0.7; // seconds per gait cycle reference

/** Joint definitions: limits, torque cap and how actions map to target angles. */
export const JOINTS = [
  { name: 'hip', min: -1.6, max: 1.4, maxTorque: 95, nominal: 0.12, range: 0.9, kp: 14 },
  { name: 'knee', min: -2.5, max: 0.02, maxTorque: 95, nominal: -0.35, range: 1.1, kp: 14 },
  { name: 'ankle', min: -0.9, max: 0.5, maxTorque: 60, nominal: 0.0, range: 0.6, kp: 14 },
];

export const LEG_NAMES = ['far', 'near'];
export const ACTIONS_PER_LEG = 3;
export const ACTION_SIZE = JOINTS.length * LEG_NAMES.length; // 6
export const OBS_SIZE = 23;

/** Reward weights (all rates, i.e. per simulated second). */
export const REWARD = {
  /** Paid per metre per second travelled. This is the objective: go right, fast. */
  forward: 2.6,
  /** Penalty per radian^2 of trunk lean (both directions). */
  upright: 0.8,
  /** Penalty per metre of squatting below the standing hip height. */
  squat: 3.0,
  /** Penalty for large actions, per action per second. */
  effort: 0.05,
  /** Penalty for changing actions (smoothness), per action per second. */
  smooth: 0.12,
  /**
   * Bonus per second of "flight": both feet off the ground while moving forward.
   * Without this, the easiest way to earn forward reward is to shuffle with both
   * feet dragging, which is a poor local optimum the policy finds immediately.
   */
  flight: 0.9,
  /** Small bonus for simply staying upright. Kept low so shuffling is not enough. */
  alive: 0.15,
  /** One-off penalty for falling. */
  fall: 10.0,
};
/** Forward speed at which the flight bonus is fully paid (m/s). */
export const FLIGHT_SPEED = 1.6;

/** Height of the fallen-enough threshold, measured at the hip pivot in metres. */
export const MIN_HIP_HEIGHT = 0.5;
export const MAX_EPISODE_TIME = 15;
export const FALL_ANGLE = 1.0;

let envCounter = 0;

/**
 * Position a body so that the point (authoredX, authoredY) of its authored frame
 * lands exactly on (worldX, worldY).
 *
 * This matters: a body's `x`/`y` is its centre of mass, which for a limb capsule
 * sits in the middle of the capsule, not at its joint pivot. Placing limbs by
 * their pivots keeps every joint's two anchors coincident at construction, so the
 * solver never has to yank them together on the first step.
 */
export function placeBody(body, authoredX, authoredY, worldX, worldY, angle = 0) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const lx = authoredX - body.comOffset.x;
  const ly = authoredY - body.comOffset.y;
  body.angle = angle;
  body.x = worldX - (c * lx - s * ly);
  body.y = worldY - (s * lx + c * ly);
  return body;
}

/**
 * Build the biped inside `world` and return its parts. Shapes are authored in a
 * convenient frame (feet soles at y = 0, hip pivot at y = LEG_LENGTH), which is
 * why joint anchors go through the authored-frame helpers.
 */
function createBiped(world, opts = {}) {
  const thighLen = opts.thighLength ?? 0.42;
  const shinLen = opts.shinLength ?? 0.40;
  const footRadius = 0.035;
  const ankleHeight = 0.07; // ankle pivot above the sole
  const hipHeight = thighLen + shinLen + ankleHeight;

  const palette = opts.palette ?? null;
  const colorOf = (key, fallback) => (palette && palette[key]) || fallback;

  const trunk = new Body({
    mass: 20,
    name: 'trunk',
    group: 'trunk',
    color: colorOf('trunk', '#3f7fd0'),
    friction: 0.8,
    shapes: [
      { a: [0, 0], b: [0, 0.44], r: 0.11 },
      { a: [0, 0.5], b: [0, 0.6], r: 0.1 },
    ],
  });
  world.addBody(trunk);
  placeBody(trunk, 0, 0, 0, hipHeight);

  const legs = [];
  for (let l = 0; l < LEG_NAMES.length; l++) {
    const side = LEG_NAMES[l];
    const xOffset = (l === 0 ? -0.035 : 0.035);
    const shade = l === 0 ? colorOf('legFar', '#2a5f9e') : colorOf('legNear', '#5aa0e8');

    const thigh = world.addBody(
      new Body({
        mass: 3.2,
        name: `thigh_${side}`,
        group: side,
        color: shade,
        friction: 0.9,
        shapes: [{ a: [0, 0], b: [0, -thighLen], r: 0.075 }],
      }),
    );
    placeBody(thigh, 0, 0, xOffset, hipHeight);
    const shin = world.addBody(
      new Body({
        mass: 2.2,
        name: `shin_${side}`,
        group: side,
        color: shade,
        friction: 0.9,
        shapes: [{ a: [0, 0], b: [0, -shinLen], r: 0.06 }],
      }),
    );
    placeBody(shin, 0, 0, xOffset, hipHeight - thighLen);
    const foot = world.addBody(
      new Body({
        mass: 0.9,
        name: `foot_${side}`,
        group: side,
        color: colorOf('foot', '#f0b429'),
        friction: 1.4,
        shapes: [{ a: [-0.13, 0], b: [0.11, 0], r: footRadius }],
      }),
    );

    placeBody(foot, 0, ankleHeight - footRadius, xOffset, hipHeight - thighLen - shinLen);

    // Joints. Anchors are given in the authored frame of the child body (the one
    // that is convenient to reason about) and converted to the world by the joint.
    const makeJoint = (index, parent, child, anchorInChild) => {
      const spec = JOINTS[index];
      const anchor = child.toWorldA(anchorInChild[0], anchorInChild[1]);
      return world.addJoint(
        new RevoluteJoint(parent, child, anchor, {
          min: spec.min,
          max: spec.max,
          maxTorque: spec.maxTorque,
          kp: spec.kp,
          damping: 0.15,
          name: `${spec.name}_${side}`,
        }),
      );
    };

    const joints = [
      makeJoint(0, trunk, thigh, [0, 0]), // hip: trunk bottom -> thigh top
      makeJoint(1, thigh, shin, [0, 0]), // knee: thigh bottom -> shin top
      makeJoint(2, shin, foot, [0, ankleHeight - footRadius]), // ankle: shin bottom -> foot pivot
    ];

    legs.push({ side, thigh, shin, foot, joints, xOffset, thighLen, shinLen });
  }

  return { trunk, legs, hipHeight, footRadius, ankleHeight };
}

export class RunningEnv {
  /**
   * @param {{seed?: number, randomize?: boolean, maxEpisodeTime?: number}} opts
   */
  constructor(opts = {}) {
    this.seed = opts.seed ?? 1;
    this.rng = mulberry32(this.seed);
    this.randomize = opts.randomize ?? true;
    this.maxEpisodeTime = opts.maxEpisodeTime ?? MAX_EPISODE_TIME;
    this.envId = ++envCounter;

    this.world = new World({ gravity: -9.81, iterations: 14, slop: 0.0015, positionIterations: 4 });
    // Static ground plane.
    this.world.addBody(
      new Body({
        mass: 1,
        isStatic: true,
        name: 'ground',
        shapes: [{ a: [-1000, -0.5], b: [1000, -0.5], r: 0.5 }],
        y: 0,
      }),
    );
    this.robot = createBiped(this.world, { palette: opts.palette });
    this.trunk = this.robot.trunk;
    this.legs = this.robot.legs;
    this.hipHeight = this.robot.hipHeight;

    this.obs = new Float64Array(OBS_SIZE);
    this.prevAction = new Float64Array(ACTION_SIZE);
    /** Hip pivot height in the settled rest pose - the reference for "upright". */
    this.restHipHeight = this.hipPivotHeight();
    this.stepCount = 0;
    this.episodeSteps = 0;
    this.time = 0;
    this.episode = 1;
    this.bestDistance = 0;
    this.lastDistance = 0;
    this.totalDistance = 0;
    this.episodesCompleted = 0;
    this.episodeReward = 0;
    this.lastEpisodeReward = 0;
    this.lastEpisodeTime = 0;
    this.terminatedByFall = false;
    this.startX = 0;
    this.reset();
  }

  /**
   * Put every body back into the start pose: a slight crouch with both feet flat
   * on the ground, optionally jittered so the policy sees a range of starts.
   */
  reset() {
    const { trunk, legs } = this;
    const rnd = (mag) => (this.randomize ? (this.rng() * 2 - 1) * mag : 0);

    const trunkAngle = rnd(0.03);
    trunk.x = 0;
    trunk.y = this.hipHeight;
    trunk.angle = trunkAngle;
    trunk.vx = rnd(0.05);
    trunk.vy = 0;
    trunk.omega = rnd(0.02);

    for (let l = 0; l < legs.length; l++) {
      const leg = legs[l];
      const hipAngle = 0.14 + rnd(0.05) + (l === 0 ? 0.03 : -0.03);
      const kneeAngle = -0.34 + rnd(0.05);
      const ankleAngle = -(hipAngle + kneeAngle) + rnd(0.03); // keeps the sole flat

      // Walk the chain joint by joint. Each joint stores its anchor in both the
      // parent's and the child's COM frame, so placing a child is exact: put the
      // child's anchor on the parent's anchor, at the angle we want.
      const angles = [trunkAngle + hipAngle, trunkAngle + hipAngle + kneeAngle];
      angles.push(angles[1] + ankleAngle);
      let parent = trunk;
      const chain = [leg.thigh, leg.shin, leg.foot];
      for (let k = 0; k < chain.length; k++) {
        const joint = leg.joints[k];
        const child = chain[k];
        const anchor = parent.toWorld(joint.localAnchorA.x, joint.localAnchorA.y);
        const c = Math.cos(angles[k]);
        const s = Math.sin(angles[k]);
        const lax = joint.localAnchorB.x;
        const lay = joint.localAnchorB.y;
        child.x = anchor.x - (c * lax - s * lay);
        child.y = anchor.y - (s * lax + c * lay);
        child.angle = angles[k];
        child.vx = trunk.vx;
        child.vy = trunk.vy;
        child.omega = trunk.omega;
        parent = child;
      }
    }

    // Settle onto the ground: shift everything so the lowest sole touches y = 0.
    this.world.contacts.length = 0;
    this.world.generateContacts();
    let lowest = Infinity;
    for (const leg of legs) {
      for (const s of leg.foot.shapes) {
        const a = leg.foot.toWorld(s.a[0], s.a[1]);
        const b = leg.foot.toWorld(s.b[0], s.b[1]);
        lowest = Math.min(lowest, a.y - s.r, b.y - s.r);
      }
    }
    const lift = -lowest;
    for (const body of this.world.bodies) {
      if (body.isStatic) continue;
      body.y += lift;
    }

    for (const j of this.world.joints) j.impulse.x = j.impulse.y = 0;
    this.world.cache.clear();
    this.world.time = 0;

    this.stepCount = 0;
    this.episodeSteps = 0;
    this.time = 0;
    this.prevAction.fill(0);
    this.episodeReward = 0;
    this.terminatedByFall = false;
    this.startX = trunk.x;
    if (!this.randomize) this.restHipHeight = this.hipPivotHeight();
    this._observe();
  }

  /** World height of the hip pivot (the point the legs hang from). */
  hipPivotHeight() {
    const joint = this.legs[0].joints[0];
    return this.trunk.toWorld(joint.localAnchorA.x, joint.localAnchorA.y).y;
  }

  /** Joint angles in action order [hipFar, kneeFar, ankleFar, hipNear, kneeNear, ankleNear]. */
  jointAngles() {
    const out = [];
    for (const leg of this.legs) for (const j of leg.joints) out.push(j.relativeAngle());
    return out;
  }

  /** Foot contact flags in leg order. */
  footContacts() {
    return this.legs.map((leg) => (this.world.isTouching(leg.foot) ? 1 : 0));
  }

  /** Fill and return the observation vector. */
  _observe() {
    const o = this.obs;
    const t = this.trunk;
    let i = 0;
    o[i++] = clamp(t.angle, -Math.PI, Math.PI);
    o[i++] = Math.sin(t.angle);
    o[i++] = Math.cos(t.angle);
    o[i++] = clamp(t.vx / 3, -3, 3);
    o[i++] = clamp(t.vy / 3, -3, 3);
    o[i++] = clamp(t.omega / 4, -3, 3);
    o[i++] = clamp(this.hipPivotHeight() / 1.0, -1, 2);
    for (const leg of this.legs) {
      for (let k = 0; k < leg.joints.length; k++) {
        const j = leg.joints[k];
        const spec = JOINTS[k];
        const scale = k === 0 ? 1.5 : k === 1 ? 2.5 : 1.0;
        o[i++] = clamp(j.relativeAngle() / scale, -3, 3);
        o[i++] = clamp(j.relativeSpeed() / 10, -3, 3);
      }
    }
    for (const leg of this.legs) o[i++] = this.world.isTouching(leg.foot) ? 1 : 0;
    const phase = (this.time % CLOCK_PERIOD) / CLOCK_PERIOD;
    o[i++] = Math.sin(2 * Math.PI * phase);
    o[i++] = Math.cos(2 * Math.PI * phase);
    return o;
  }

  /** Current phase clock values (used by the renderer). */
  clock() {
    const phase = (this.time % CLOCK_PERIOD) / CLOCK_PERIOD;
    return [Math.sin(2 * Math.PI * phase), Math.cos(2 * Math.PI * phase)];
  }

  /**
   * Apply an action (array of ACTION_SIZE values in [-1, 1]) and advance one
   * control step (SUBSTEPS physics substeps). Returns {obs, reward, terminated,
   * truncated, info}.
   */
  step(action) {
    const { world } = this;

    // 1. Actions -> motor targets.
    let ai = 0;
    for (const leg of this.legs) {
      for (let k = 0; k < leg.joints.length; k++) {
        const spec = JOINTS[k];
        const a = clamp(action[ai], -1, 1);
        leg.joints[k].target = clamp(spec.nominal + a * spec.range, spec.min, spec.max);
        ai++;
      }
    }

    // 2. Physics.
    for (let s = 0; s < SUBSTEPS; s++) world.step(PHYSICS_DT);

    this.stepCount++;
    this.episodeSteps++;
    this.time += CONTROL_DT;

    // 3. Reward.
    const t = this.trunk;
    const vx = t.vx;
    const uprightPenalty = REWARD.upright * t.angle * t.angle;
    const squatPenalty = REWARD.squat * Math.max(0, this.restHipHeight - this.hipPivotHeight());
    let effort = 0;
    let smooth = 0;
    ai = 0;
    for (const leg of this.legs) {
      for (let k = 0; k < leg.joints.length; k++) {
        const a = clamp(action[ai], -1, 1);
        effort += a * a;
        const d = a - this.prevAction[ai];
        smooth += d * d;
        this.prevAction[ai] = a;
        ai++;
      }
    }
    effort *= REWARD.effort / ACTION_SIZE;
    smooth *= REWARD.smooth / ACTION_SIZE;

    const contacts = this.footContacts();
    const airborne = contacts[0] === 0 && contacts[1] === 0 && vx > 0.3;
    const flightBonus = airborne ? REWARD.flight * clamp(vx / FLIGHT_SPEED, 0, 1) * CONTROL_DT : 0;

    let reward =
      REWARD.forward * clamp(vx, -1, 8) * CONTROL_DT +
      REWARD.alive * CONTROL_DT +
      flightBonus -
      (uprightPenalty + squatPenalty + effort + smooth) * CONTROL_DT;

    // 4. Termination.
    let terminated = false;
    let truncated = false;
    if (Math.abs(t.angle) > FALL_ANGLE || this.hipPivotHeight() < MIN_HIP_HEIGHT || world.isTouching(t)) {
      terminated = true;
      reward -= REWARD.fall;
    } else if (this.time >= this.maxEpisodeTime) {
      truncated = true;
    }

    const info = {
      x: t.x - this.startX,
      vx,
      time: this.time,
      contacts,
      airborne,
      jointAngles: this.jointAngles(),
    };
    this.episodeReward += reward;
    info.episodeReward = this.episodeReward;

    if (terminated || truncated) {
      const distance = Math.max(0, t.x - this.startX);
      this.lastDistance = distance;
      this.totalDistance += distance;
      this.bestDistance = Math.max(this.bestDistance, distance);
      this.lastEpisodeReward = this.episodeReward;
      this.lastEpisodeTime = this.time;
      this.episodesCompleted++;
      this.episode++;
      this.terminatedByFall = terminated;
      // Fast path: reset in place, keep the same bodies.
      this.reset();
    } else {
      this._observe();
    }

    return { obs: this.obs, reward, terminated, truncated, info };
  }

  /**
   * Renderable geometry: world space capsule segments, plus the contact flags.
   * Used by the browser client, which receives this instead of the raw bodies.
   */
  snapshot() {
    const bodies = [];
    for (const b of this.world.bodies) {
      if (b.isStatic) continue;
      const shapes = b.shapes.map((s) => {
        const a = b.toWorld(s.a[0], s.a[1]);
        const bb = b.toWorld(s.b[0], s.b[1]);
        return [round3(a.x), round3(a.y), round3(bb.x), round3(bb.y), round3(s.r)];
      });
      bodies.push({ n: b.name, g: b.group || b.name, x: round3(b.x), y: round3(b.y), a: round3(b.angle), s: shapes });
    }
    return {
      bodies,
      contacts: this.footContacts(),
      x: round3(this.trunk.x - this.startX),
      totalX: round3(this.trunk.x),
      t: round3(this.time),
      episode: this.episode,
      vx: round2(this.trunk.vx),
      hip: round3(this.hipPivotHeight()),
    };
  }
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}
function round2(v) {
  return Math.round(v * 100) / 100;
}
