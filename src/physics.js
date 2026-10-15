/**
 * A small planar (2D) rigid-body physics engine, written from scratch.
 *
 * Features
 *  - Rigid bodies whose collision geometry is a set of capsules (line segment + radius)
 *  - Revolute (pin) joints solved as 2D point-to-point velocity constraints
 *  - Angular joint limits (one sided constraints with accumulated impulses)
 *  - Joint motors solved as velocity servos with a torque cap (unconditionally stable,
 *    which matters because the RL policy pushes them hard)
 *  - Contacts against the ground plane y = 0 with Coulomb friction, Baumgarte
 *    position bias and warm starting across substeps
 *
 * Everything is plain arithmetic on object fields: no allocations inside `step`
 * except the per-substep contact list, which is reused. The engine is fast enough
 * to run thousands of simulated agent-seconds per second of CPU time.
 */
import { clamp, cross } from './math.js';

const EPS = 1e-9;

/** Axis aligned capsule shape in body-local coordinates. */
function makeShape(s) {
  return {
    a: [s.a[0], s.a[1]],
    b: [s.b[0], s.b[1]],
    r: s.r,
    /** 0 = not touching ground, 1 = touching (refreshed every substep). */
    touching: 0,
  };
}

/**
 * A rigid body. `shapes` are given in a convenient local frame; the constructor
 * re-centres them on the body's centre of mass (mass is distributed by area), so
 * the body frame origin is always the COM.
 */
export class Body {
  constructor({ mass, shapes, x = 0, y = 0, angle = 0, name = 'body', color = '#8899aa', isStatic = false, friction = 1.0 }) {
    this.name = name;
    this.color = color;
    this.isStatic = isStatic;
    this.friction = friction;

    const rawShapes = shapes.map(makeShape);
    // Area based mass distribution (capsule ~ rectangle + two half discs).
    const areas = rawShapes.map((s) => {
      const len = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
      return len * 2 * s.r + Math.PI * s.r * s.r;
    });
    const totalArea = areas.reduce((a, b) => a + b, 0) || 1;

    // Centre of mass in the raw (authoring) frame.
    let cx = 0;
    let cy = 0;
    for (let i = 0; i < rawShapes.length; i++) {
      const s = rawShapes[i];
      const mi = mass * (areas[i] / totalArea);
      cx += mi * 0.5 * (s.a[0] + s.b[0]);
      cy += mi * 0.5 * (s.a[1] + s.b[1]);
    }
    cx /= mass;
    cy /= mass;

    // Re-centre shapes on the COM and compute the moment of inertia.
    let inertia = 0;
    for (let i = 0; i < rawShapes.length; i++) {
      const s = rawShapes[i];
      s.a[0] -= cx;
      s.a[1] -= cy;
      s.b[0] -= cx;
      s.b[1] -= cy;
      const mi = mass * (areas[i] / totalArea);
      const len = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
      const rectArea = len * 2 * s.r;
      const discArea = Math.PI * s.r * s.r;
      const rectMass = rectArea / (rectArea + discArea) * mi;
      const discMass = mi - rectMass;
      // Rectangle about its centre (perpendicular axis) + two half discs
      // (a half disc about its flat centre is ~0.5*m*r^2 for the pair).
      const localI = rectMass * (len * len + (2 * s.r) * (2 * s.r)) / 12 + discMass * 0.5 * s.r * s.r;
      const mx = 0.5 * (s.a[0] + s.b[0]);
      const my = 0.5 * (s.a[1] + s.b[1]);
      inertia += localI + mi * (mx * mx + my * my);
    }

    this.mass = isStatic ? Infinity : mass;
    this.invMass = isStatic ? 0 : 1 / mass;
    this.inertia = isStatic ? Infinity : inertia;
    this.invInertia = isStatic ? 0 : 1 / Math.max(inertia, EPS);
    this.shapes = rawShapes;
    /**
     * Offset of the shape centroid (the true centre of mass) relative to the body
     * origin, expressed in the authored frame. Shapes are re-centred on the COM so
     * the body state (x, y) is always the COM, but authors write shapes and joint
     * anchors in whatever frame is convenient; `toWorldA`/`toLocalA` translate
     * between that authored frame and the world.
     */
    this.comOffset = { x: cx, y: cy };

    this.x = x;
    this.y = y;
    this.angle = angle;
    this.vx = 0;
    this.vy = 0;
    this.omega = 0;
    /** Largest normal impulse applied to this body during the last substep. */
    this.contactImpulse = 0;
  }

  /** Local point -> world point. */
  toWorld(lx, ly) {
    const c = Math.cos(this.angle);
    const s = Math.sin(this.angle);
    return { x: this.x + c * lx - s * ly, y: this.y + s * lx + c * ly };
  }

  /** World point -> local point. */
  toLocal(wx, wy) {
    const c = Math.cos(this.angle);
    const s = Math.sin(this.angle);
    const dx = wx - this.x;
    const dy = wy - this.y;
    return { x: c * dx + s * dy, y: -s * dx + c * dy };
  }

  /**
   * Authored frame -> world. Use this when placing joint anchors and motor limits
   * against the same coordinates the shapes were written in.
   */
  toWorldA(lx, ly) {
    return this.toWorld(lx - this.comOffset.x, ly - this.comOffset.y);
  }

  /** World -> authored frame. */
  toLocalA(wx, wy) {
    const p = this.toLocal(wx, wy);
    return { x: p.x + this.comOffset.x, y: p.y + this.comOffset.y };
  }

  /** Velocity of a point at world offset (rx, ry) from the COM. */
  velocityAt(rx, ry) {
    return { x: this.vx - this.omega * ry, y: this.vy + this.omega * rx };
  }

  /** Apply an impulse at world offset (rx, ry) from the COM. */
  applyImpulse(jx, jy, rx, ry) {
    if (this.isStatic) return;
    this.vx += jx * this.invMass;
    this.vy += jy * this.invMass;
    this.omega += this.invInertia * cross(rx, ry, jx, jy);
  }

  /** Apply an angular impulse. */
  applyAngularImpulse(j) {
    if (this.isStatic) return;
    this.omega += j * this.invInertia;
  }
}

/** Revolute joint with optional limits and a velocity-servo motor. */
export class RevoluteJoint {
  /**
   * @param {Body} bodyA parent body (the joint angle is measured as B - A)
   * @param {Body} bodyB child body
   * @param {{x:number,y:number}} anchorWorld shared anchor point
   */
  constructor(bodyA, bodyB, anchorWorld, opts = {}) {
    this.bodyA = bodyA;
    this.bodyB = bodyB;
    this.localAnchorA = bodyA.toLocal(anchorWorld.x, anchorWorld.y);
    this.localAnchorB = bodyB.toLocal(anchorWorld.x, anchorWorld.y);
    this.name = opts.name || `${bodyA.name}->${bodyB.name}`;
    this.min = opts.min ?? -Math.PI;
    this.max = opts.max ?? Math.PI;
    this.maxTorque = opts.maxTorque ?? 0;
    this.maxSpeed = opts.maxSpeed ?? 12;
    /** P gain of the position->velocity servo (1/s). */
    this.kp = opts.kp ?? 12;
    this.target = opts.target ?? 0;
    /** Extra viscous damping torques (N m s / rad). */
    this.damping = opts.damping ?? 0.02;
    /** Accumulated impulses (solver state). */
    this.impulse = { x: 0, y: 0 };
    this.limitImpulse = 0;
    this.motorImpulse = 0;
    this.angle = 0;
  }

  /** Relative joint angle in radians. */
  relativeAngle() {
    return this.bodyB.angle - this.bodyA.angle;
  }

  /** Angular velocity of B relative to A. */
  relativeSpeed() {
    return this.bodyB.omega - this.bodyA.omega;
  }

  /** Maximum angular impulse the motor may deliver in a substep. */
  maxMotorImpulse(dt) {
    return this.maxTorque * dt;
  }
}

export class World {
  constructor(opts = {}) {
    this.gravity = opts.gravity ?? -9.81;
    this.iterations = opts.iterations ?? 16;
    this.baumgarte = opts.baumgarte ?? 0.2;
    this.slop = opts.slop ?? 0.0015;
    /**
     * Penetration is removed by projecting positions after the velocity solve
     * ("split impulse"). Doing it at the velocity level instead would hand the
     * bodies free kinetic energy and make resting feet bounce off the ground.
     */
    this.positionIterations = opts.positionIterations ?? 4;
    this.positionBaumgarte = opts.positionBaumgarte ?? 0.2;
    this.maxCorrection = opts.maxCorrection ?? 0.2;
    this.bodies = [];
    this.joints = [];
    /** @type {Array<{body: Body, shapeIndex: number, px: number, py: number, nx: number, ny: number, penetration: number, jn: number, jt: number, rx: number, ry: number, key: string}>} */
    this.contacts = [];
    this.cache = new Map();
    this.time = 0;
    this.steps = 0;
  }

  addBody(body) {
    this.bodies.push(body);
    body.index = this.bodies.length - 1;
    return body;
  }

  addJoint(joint) {
    this.joints.push(joint);
    return joint;
  }

  /**
   * Number of ground contacts for `body`. Contacts only exist while a shape
   * actually penetrates the ground, so this is a reliable "is a foot down" test.
   */
  touchCount(body) {
    let n = 0;
    for (const c of this.contacts) {
      if (c.body === body && c.jn > 0) n++;
    }
    return n;
  }

  /** True when any shape of `body` is in contact with the ground. */
  isTouching(body) {
    return this.touchCount(body) > 0;
  }

  /** Advance the simulation by one substep. */
  step(dt) {
    this.time += dt;
    this.steps++;

    // 1. Integrate velocities (gravity) and apply joint damping + motor targets.
    for (const b of this.bodies) {
      if (b.isStatic) continue;
      b.vy += this.gravity * dt;
      b.contactImpulse = 0;
    }
    for (const j of this.joints) {
      if (j.damping > 0) {
        const w = j.relativeSpeed();
        const t = -j.damping * w;
        j.bodyB.applyAngularImpulse(t * dt);
        j.bodyA.applyAngularImpulse(-t * dt);
      }
      // Motors and limits are 1-DOF constraints solved from scratch every substep:
      // a fresh impulse budget each substep is what lets a saturated motor keep
      // delivering its full rated torque instead of running out of accumulator.
      j.motorImpulse = 0;
      j.limitImpulse = 0;
    }

    // 2. Joint anchors for this substep.
    for (const j of this.joints) {
      const a = j.bodyA.toWorld(j.localAnchorA.x, j.localAnchorA.y);
      const b = j.bodyB.toWorld(j.localAnchorB.x, j.localAnchorB.y);
      j.worldA = a;
      j.worldB = b;
      j.errorX = b.x - a.x;
      j.errorY = b.y - a.y;
      j.rAx = a.x - j.bodyA.x;
      j.rAy = a.y - j.bodyA.y;
      j.rBx = b.x - j.bodyB.x;
      j.rBy = b.y - j.bodyB.y;
      j.angle = j.bodyB.angle - j.bodyA.angle;
    }

    // 3. Contacts against the ground plane.
    this.generateContacts();

    // 4. Warm start.
    for (const c of this.contacts) {
      const nImp = c.jn;
      const tImp = c.jt;
      c.body.applyImpulse(c.nx * nImp + c.tx * tImp, c.ny * nImp + c.ty * tImp, c.rx, c.ry);
    }
    for (const j of this.joints) {
      const p = j.impulse;
      j.bodyA.applyImpulse(-p.x, -p.y, j.rAx, j.rAy);
      j.bodyB.applyImpulse(p.x, p.y, j.rBx, j.rBy);
    }

    // 5. Solve velocity constraints.
    for (let iter = 0; iter < this.iterations; iter++) {
      for (const j of this.joints) this.solveMotor(j, dt);
      for (const c of this.contacts) this.solveContact(c, dt);
      for (const j of this.joints) this.solveJoint(j, dt);
      for (const j of this.joints) this.solveLimit(j, dt);
    }
    this.storeContactImpulses();

    // 6. Integrate positions.
    for (const b of this.bodies) {
      if (b.isStatic) continue;
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.angle += b.omega * dt;
    }

    // 7. Repair penetration by projection (no energy added).
    this.solvePositions();

    // 8. Reject non-finite states before they can poison the whole run.
    for (const b of this.bodies) {
      if (b.isStatic) continue;
      if (!Number.isFinite(b.x) || !Number.isFinite(b.y) || !Number.isFinite(b.angle)) {
        throw new Error(`physics diverged for body "${b.name}" at t=${this.time.toFixed(3)}s`);
      }
    }
  }

  /**
   * Build the ground contact list for this substep: every capsule endpoint below
   * y = 0 becomes a contact. Impulses are warm started from the previous substep
   * (matched by body/shape/endpoint key) which is what keeps standing feet from
   * sinking or buzzing under load.
   */
  generateContacts() {
    const cached = this.cache;
    const used = this.usedKeys || (this.usedKeys = new Set());
    used.clear();
    let n = 0;
    for (const b of this.bodies) {
      if (b.isStatic) continue;
      for (let si = 0; si < b.shapes.length; si++) {
        const s = b.shapes[si];
        s.touching = 0;
        const len = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
        // A degenerate capsule (a circle) has a single contact point, not two.
        const ends = len < 1e-6 ? 1 : 2;
        const wa = b.toWorld(s.a[0], s.a[1]);
        const wb = ends === 2 ? b.toWorld(s.b[0], s.b[1]) : wa;
        for (let e = 0; e < ends; e++) {
          const wx = e === 0 ? wa.x : wb.x;
          const wy = e === 0 ? wa.y : wb.y;
          const penetration = s.r - wy;
          if (penetration <= 0) continue;
          const key = `${b.index}|${si}|${e}`;
          let rec = cached.get(key);
          if (!rec) {
            rec = { jn: 0, px: wx };
            cached.set(key, rec);
          }
          used.add(key);
          // A contact that slid far away is a different contact: forget its impulse.
          if (Math.abs(rec.px - wx) > 0.5) rec.jn = 0;
          rec.px = wx;
          s.touching = 1;

          const c = this.contacts[n] || (this.contacts[n] = {});
          c.body = b;
          c.shapeIndex = si;
          c.end = e;
          c.rec = rec;
          // Only the normal impulse is warm started. Warm starting the tangential
          // impulse would make it collapse to the fresh friction budget whenever the
          // normal impulse dips, and that collapse is itself an impulse - it kicks
          // resting bodies. Friction converges fine within a substep on its own.
          c.jn = rec.jn;
          c.jt = 0;
          c.nx = 0;
          c.ny = 1;
          c.tx = 1;
          c.ty = 0;
          c.penetration = penetration;
          c.rx = wx - b.x;
          c.ry = -b.y;
          // Local endpoint + radius, so the position solver can re-derive the
          // penetration after the bodies have moved.
          c.lx = e === 0 ? s.a[0] : s.b[0];
          c.ly = e === 0 ? s.a[1] : s.b[1];
          c.radius = s.r;
          n++;
        }
      }
    }
    this.contacts.length = n;
    // Drop impulses whose contacts have gone away.
    for (const key of cached.keys()) {
      if (!used.has(key)) cached.delete(key);
    }
  }

  /** Friction + normal impulse for one contact point. */
  solveContact(c, dt) {
    const b = c.body;
    if (b.invMass === 0) return;
    // `velocityAt` already includes the rotational term (omega x r): the effective
    // mass below is derived for the *point* velocity, so the angular contribution
    // must not be added a second time here.
    const v = b.velocityAt(c.rx, c.ry);

    // --- normal: the point may not approach the ground (no push-out term here,
    //     penetration is repaired later by the position solver) ----------------
    const rn = cross(c.rx, c.ry, c.nx, c.ny);
    const kn = b.invMass + b.invInertia * rn * rn;
    const vn = v.x * c.nx + v.y * c.ny;
    let dJn = -vn / kn;
    const oldJn = c.jn;
    c.jn = Math.max(0, oldJn + dJn);
    dJn = c.jn - oldJn;

    // --- friction: tangential impulse capped by Coulomb's law ----------------
    const rt = cross(c.rx, c.ry, c.tx, c.ty);
    const kt = b.invMass + b.invInertia * rt * rt;
    const vt = v.x * c.tx + v.y * c.ty;
    let dJt = -vt / kt;
    const maxFriction = b.friction * c.jn;
    const oldJt = c.jt;
    c.jt = clamp(oldJt + dJt, -maxFriction, maxFriction);
    dJt = c.jt - oldJt;

    b.applyImpulse(c.nx * dJn + c.tx * dJt, c.ny * dJn + c.ty * dJt, c.rx, c.ry);
    if (c.jn > b.contactImpulse) b.contactImpulse = c.jn;
  }

  /**
   * Position projection: push penetrating contact points out of the ground by
   * moving the bodies (not by speeding them up). Runs after the velocity solve
   * and position integration, a few iterations per substep.
   */
  solvePositions() {
    let maxPenetration = 0;
    for (let iter = 0; iter < this.positionIterations; iter++) {
      let worst = 0;
      for (const c of this.contacts) {
        const b = c.body;
        if (b.invMass === 0) continue;
        const world = b.toWorld(c.lx, c.ly);
        const penetration = c.radius - world.y;
        if (penetration <= 0) continue;
        const correction = Math.min(this.maxCorrection, this.positionBaumgarte * Math.max(0, penetration - this.slop));
        if (correction <= 0) continue;
        const rx = world.x - b.x;
        const ry = world.y - b.y;
        const rn = cross(rx, ry, c.nx, c.ny);
        const k = b.invMass + b.invInertia * rn * rn;
        if (k < EPS) continue;
        const impulse = correction / k;
        b.x += b.invMass * impulse * c.nx;
        b.y += b.invMass * impulse * c.ny;
        b.angle += b.invInertia * cross(rx, ry, impulse * c.nx, impulse * c.ny);
        worst = Math.max(worst, penetration);
      }
      if (worst > maxPenetration) maxPenetration = worst;
      if (worst <= this.slop) break;
    }
    return maxPenetration;
  }

  /** Copy accumulated normal impulses back into the warm start cache. */
  storeContactImpulses() {
    for (const c of this.contacts) {
      c.rec.jn = c.jn;
    }
  }

  /** Point-to-point constraint holding the two joint anchors together. */
  solveJoint(j, dt) {
    const A = j.bodyA;
    const B = j.bodyB;
    const invMA = A.invMass;
    const invMB = B.invMass;
    if (invMA === 0 && invMB === 0) return;

    const iA = A.invInertia;
    const iB = B.invInertia;
    const rAx = j.rAx;
    const rAy = j.rAy;
    const rBx = j.rBx;
    const rBy = j.rBy;

    const k11 = invMA + invMB + iA * rAy * rAy + iB * rBy * rBy;
    const k12 = -iA * rAx * rAy - iB * rBx * rBy;
    const k22 = invMA + invMB + iA * rAx * rAx + iB * rBx * rBx;
    const det = k11 * k22 - k12 * k12;
    if (Math.abs(det) < EPS) return;
    const invDet = 1 / det;

    // Relative velocity at the anchors plus a positional bias.
    const va = A.velocityAt(rAx, rAy);
    const vb = B.velocityAt(rBx, rBy);
    const beta = this.baumgarte / dt;
    const cx = vb.x - va.x + beta * j.errorX;
    const cy = vb.y - va.y + beta * j.errorY;

    const px = -invDet * (k22 * cx - k12 * cy);
    const py = -invDet * (k11 * cy - k12 * cx);

    j.impulse.x += px;
    j.impulse.y += py;
    A.applyImpulse(-px, -py, rAx, rAy);
    B.applyImpulse(px, py, rBx, rBy);
  }

  /** Velocity servo: drive the joint towards `target` at up to `maxSpeed`, capped by torque. */
  solveMotor(j, dt) {
    if (j.maxTorque <= 0) return;
    const A = j.bodyA;
    const B = j.bodyB;
    const invI = A.invInertia + B.invInertia;
    if (invI < EPS) return;

    const error = j.target - j.angle;
    const desired = clamp(error * j.kp, -j.maxSpeed, j.maxSpeed);
    const current = B.omega - A.omega;
    let impulse = (desired - current) / invI;
    const maxImpulse = j.maxMotorImpulse(dt);
    const old = j.motorImpulse;
    j.motorImpulse = clamp(old + impulse, -maxImpulse, maxImpulse);
    impulse = j.motorImpulse - old;
    B.applyAngularImpulse(impulse);
    A.applyAngularImpulse(-impulse);
  }

  /**
   * One sided joint limits: a hard stop with a small slop and a Baumgarte push-out,
   * so a knee bent to its limit is held at the limit instead of slowly creeping past
   * it the way a pure velocity clamp would.
   */
  solveLimit(j, dt) {
    const A = j.bodyA;
    const B = j.bodyB;
    const invI = A.invInertia + B.invInertia;
    if (invI < EPS) return;
    const angle = B.angle - A.angle;
    const speed = B.omega - A.omega;
    const beta = this.baumgarte / dt;
    const slop = 0.005;

    const old = j.limitImpulse;
    let next = 0;
    if (angle > j.max) {
      // Push the joint angle back down: angular impulse must be negative.
      const bias = beta * Math.max(0, angle - j.max - slop);
      next = Math.min(0, old + (-(speed + bias) / invI));
    } else if (angle < j.min) {
      // Push the joint angle back up: angular impulse must be positive.
      const bias = beta * Math.max(0, j.min - angle - slop);
      next = Math.max(0, old + ((bias - speed) / invI));
    }
    const impulse = next - old;
    j.limitImpulse = next;
    if (impulse !== 0) {
      B.applyAngularImpulse(impulse);
      A.applyAngularImpulse(-impulse);
    }
  }
}
