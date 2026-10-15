/**
 * Neural network and optimisation primitives, written from scratch.
 *
 * There is no autodiff here on purpose: the network is two dense layers with
 * tanh activations plus two linear heads, so the backward pass is short enough
 * to write by hand - and the test suite gradient-checks it against finite
 * differences, which is the only way to trust a hand-written backward pass.
 *
 * Everything is Float64Array, contiguous and row-major by sample, which is what
 * makes plain loops reach ~1 GFLOP/s in the JS JIT (measured on this machine).
 */
import { gaussian, randInt, clamp } from './math.js';

/** A dense layer: y = activation(x @ W + b). */
export class Dense {
  /**
   * @param {number} nin
   * @param {number} nout
   * @param {() => number} rng
   * @param {{activation?: 'tanh'|'linear', gain?: number}} opts
   */
  constructor(nin, nout, rng, opts = {}) {
    this.nin = nin;
    this.nout = nout;
    this.activation = opts.activation ?? 'tanh';
    const gain = opts.gain ?? 1;
    // Scaled Gaussian init keeps pre-activations O(1) for tanh layers.
    const scale = (gain / Math.sqrt(nin)) * Math.sqrt(2);
    this.W = new Float64Array(nin * nout);
    for (let i = 0; i < this.W.length; i++) this.W[i] = gaussian(rng) * scale;
    this.b = new Float64Array(nout);
    this.gW = new Float64Array(nin * nout);
    this.gb = new Float64Array(nout);
  }

  /** Grow the scratch buffers so they can hold a batch of `B` samples. */
  _ensure(B) {
    const needPre = B * this.nout;
    const needDX = B * this.nin;
    if (!this._pre || this._pre.length < needPre) {
      this._pre = new Float64Array(needPre);
      this._dX = new Float64Array(needDX);
    }
    if (!this._out || this._out.length < this.nout * B) this._out = new Float64Array(this.nout * B);
  }

  /**
   * Forward for a batch of B samples. X is B x nin row-major.
   *
   * The returned array is an internal scratch buffer reused by the next forward
   * call on this layer (which is what keeps the PPO update allocation-free), so
   * callers must not hold on to it across another forward pass.
   *
   * @returns {Float64Array} B x nout output view
   */
  forward(X, B) {
    const { nin, nout } = this;
    this._ensure(B);
    const pre = this._pre.subarray(0, B * nout);
    const out = this._out.subarray(0, B * nout);
    pre.fill(0);
    for (let j = 0; j < nin; j++) {
      for (let m = 0; m < nout; m++) {
        const w = this.W[j * nout + m];
        if (w === 0) continue;
        for (let i = 0; i < B; i++) pre[i * nout + m] += X[i * nin + j] * w;
      }
    }
    for (let i = 0; i < B; i++) {
      for (let m = 0; m < nout; m++) {
        const idx = i * nout + m;
        const z = pre[idx] + this.b[m];
        out[idx] = this.activation === 'tanh' ? Math.tanh(z) : z;
      }
    }
    this._cache = { X, B, pre, out };
    return out;
  }

  /**
   * Accumulate gradients for a batch and return dX (B x nin) so that earlier
   * layers can continue the chain. Gradients accumulate, so the caller zeroes
   * them once per optimiser step. The returned array is this layer's scratch
   * buffer and is valid until the next backward call on the same layer.
   */
  backward(dOut, X, B) {
    const { nin, nout } = this;
    this._ensure(B);
    const pre = this._cache.pre;
    const dPre = this._pre.subarray(0, B * nout);
    const dX = this._dX.subarray(0, B * nin);
    for (let i = 0; i < B; i++) {
      for (let m = 0; m < nout; m++) {
        const idx = i * nout + m;
        if (this.activation === 'tanh') {
          const y = Math.tanh(pre[idx] + this.b[m]);
          dPre[idx] = dOut[idx] * (1 - y * y);
        } else {
          dPre[idx] = dOut[idx];
        }
      }
    }
    for (let j = 0; j < nin; j++) {
      for (let i = 0; i < B; i++) dX[i * nin + j] = 0;
    }
    for (let j = 0; j < nin; j++) {
      for (let m = 0; m < nout; m++) {
        const w = this.W[j * nout + m];
        const wIdx = j * nout + m;
        let acc = 0;
        for (let i = 0; i < B; i++) {
          const g = dPre[i * nout + m];
          acc += X[i * nin + j] * g;
          dX[i * nin + j] += g * w;
        }
        this.gW[wIdx] += acc;
      }
    }
    for (let m = 0; m < nout; m++) {
      let acc = 0;
      for (let i = 0; i < B; i++) acc += dPre[i * nout + m];
      this.gb[m] += acc;
    }
    return dX;
  }

  zeroGrad() {
    this.gW.fill(0);
    this.gb.fill(0);
  }

  /** Parameter/gradient views for the optimiser. */
  collectParams(params, grads) {
    params.push(this.W, this.b);
    grads.push(this.gW, this.gb);
  }
}

/**
 * Actor-critic: a shared tanh trunk with a Gaussian policy head and a value head.
 * The policy is Gaussian with a state-independent learned log standard deviation,
 * the standard parameterisation for continuous control.
 */
export class Policy {
  /**
   * @param {{obsSize:number, actSize:number, hidden?:number[], seed?:number,
   *          logStdInit?:number, outGain?:number}} opts
   */
  constructor(opts) {
    const { obsSize, actSize, hidden = [64, 64], seed = 1 } = opts;
    this.obsSize = obsSize;
    this.actSize = actSize;
    this.hidden = hidden.slice();
    // A dedicated RNG stream keeps policy initialisation reproducible and separate
    // from the environment's randomness.
    const rng = mulberry(seed * 7919 + 13);

    const sizes = [obsSize, ...hidden];
    this.trunk = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      this.trunk.push(new Dense(sizes[i], sizes[i + 1], rng, { gain: 1 }));
    }
    this.meanHead = new Dense(hidden[hidden.length - 1], actSize, rng, {
      activation: 'linear',
      gain: opts.outGain ?? 0.01,
    });
    this.valueHead = new Dense(hidden[hidden.length - 1], 1, rng, { activation: 'linear', gain: 1 });
    this.valueHead.W.fill(0);
    this.valueHead.b.fill(0);

    this.logStd = new Float64Array(actSize).fill(opts.logStdInit ?? -0.7);
    this.gLogStd = new Float64Array(actSize);

    this._mean = new Float64Array(actSize);
    this._value = new Float64Array(1);
  }

  /** Parameter views + matching gradient views, in a stable order. */
  collect() {
    const params = [];
    const grads = [];
    for (const layer of this.trunk) layer.collectParams(params, grads);
    this.meanHead.collectParams(params, grads);
    this.valueHead.collectParams(params, grads);
    params.push(this.logStd);
    grads.push(this.gLogStd);
    return { params, grads };
  }

  zeroGrad() {
    for (const layer of this.trunk) layer.zeroGrad();
    this.meanHead.zeroGrad();
    this.valueHead.zeroGrad();
    this.gLogStd.fill(0);
  }

  /** Forward one observation. Returns the mean action and the value. */
  forward(obs) {
    const B = 1;
    this._obsBuf = this._obsBuf || new Float64Array(this.obsSize);
    for (let i = 0; i < this.obsSize; i++) this._obsBuf[i] = obs[i];
    let h = this._obsBuf;
    for (const layer of this.trunk) h = layer.forward(h, B);
    // The two heads use independent scratch buffers, so reading the mean before
    // running the value head is only a matter of clarity - but copying it out
    // makes the returned mean safe to keep.
    const mean = this.meanHead.forward(h, B);
    for (let i = 0; i < this.actSize; i++) this._mean[i] = mean[i];
    const value = this.valueHead.forward(h, B);
    return { mean: this._mean, value: value[0] };
  }

  /** Standard deviations of the action distribution (clamped for stability). */
  std() {
    const out = new Float64Array(this.actSize);
    for (let i = 0; i < this.actSize; i++) {
      const ls = clamp(this.logStd[i], -4, 1);
      out[i] = Math.exp(ls);
    }
    return out;
  }

  /**
   * Sample an action for `obs`. Tanh-squashed? No: actions are clipped to [-1, 1]
   * in the environment, and the Gaussian is centred on a bounded target range, so
   * sampling stays in the useful region without a squashing Jacobian.
   * @returns {{action: Float64Array, logProb: number, value: number, mean: Float64Array, std: Float64Array, sample: Float64Array}}
   */
  act(obs, rng) {
    const { mean, value } = this.forward(obs);
    const std = this.std();
    const action = new Float64Array(this.actSize);
    const sample = new Float64Array(this.actSize);
    let logProb = 0;
    for (let i = 0; i < this.actSize; i++) {
      const z = gaussian(rng);
      sample[i] = z;
      const a = mean[i] + std[i] * z;
      action[i] = a;
      const dev = (a - mean[i]) / std[i];
      logProb += -0.5 * dev * dev - Math.log(std[i]) - 0.5 * Math.log(2 * Math.PI);
    }
    return { action, logProb, value, mean: Float64Array.from(mean), std, sample };
  }

  /** Mean action only (deterministic evaluation, used for the champion run). */
  actDeterministic(obs) {
    const { mean } = this.forward(obs);
    return Float64Array.from(mean);
  }

  /** Serialise to a compact, human-inspectable JSON object. */
  toJSON() {
    const blob = (arr) => {
      const f32 = new Float32Array(arr);
      const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
      let bin = '';
      for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
      return typeof btoa === 'function' ? btoa(bin) : Buffer.from(bytes).toString('base64');
    };
    return {
      format: 'rl-running-policy@1',
      obsSize: this.obsSize,
      actSize: this.actSize,
      hidden: this.hidden,
      trunk: this.trunk.map((l) => ({ w: blob(l.W), b: blob(l.b), nin: l.nin, nout: l.nout, act: l.activation })),
      meanHead: { w: blob(this.meanHead.W), b: blob(this.meanHead.b), nin: this.meanHead.nin, nout: this.meanHead.nout },
      valueHead: { w: blob(this.valueHead.W), b: blob(this.valueHead.b), nin: this.valueHead.nin, nout: this.valueHead.nout },
      logStd: Array.from(this.logStd, (v) => Math.round(v * 1e4) / 1e4),
    };
  }

  /** Rebuild a policy from `toJSON()` output. */
  static fromJSON(obj) {
    const unblob = (s) => {
      const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return Float64Array.from(new Float32Array(bytes.buffer));
    };
    const policy = new Policy({
      obsSize: obj.obsSize,
      actSize: obj.actSize,
      hidden: obj.hidden,
      seed: 1,
    });
    policy.trunk = obj.trunk.map((l) => {
      const layer = new Dense(l.nin, l.nout, mulberry(1), { activation: l.act ?? 'tanh' });
      layer.W.set(unblob(l.w));
      layer.b.set(unblob(l.b));
      return layer;
    });
    for (const [head, src] of [
      [policy.meanHead, obj.meanHead],
      [policy.valueHead, obj.valueHead],
    ]) {
      head.W.set(unblob(src.w));
      head.b.set(unblob(src.b));
    }
    policy.logStd.set(Float64Array.from(obj.logStd));
    return policy;
  }

  /** Copy parameters from another policy (same architecture). */
  copyFrom(other) {
    const a = this.collect();
    const b = other.collect();
    for (let i = 0; i < a.params.length; i++) a.params[i].set(b.params[i]);
  }

  /** Independent deep copy. */
  clone() {
    const copy = Policy.fromJSON(this.toJSON());
    return copy;
  }

  /** Number of trainable scalars. */
  paramCount() {
    return this.collect().params.reduce((n, p) => n + p.length, 0);
  }
}

/** Adam optimiser over a list of parameter arrays with matching gradients. */
export class Adam {
  constructor(params, grads, opts = {}) {
    this.params = params;
    this.grads = grads;
    this.lr = opts.lr ?? 3e-4;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.eps = opts.eps ?? 1e-8;
    // Small L2 decay is used as a weight decay style regulariser.
    this.weightDecay = opts.weightDecay ?? 0;
    this.m = params.map((p) => new Float64Array(p.length));
    this.v = params.map((p) => new Float64Array(p.length));
    this.t = 0;
  }

  /** Global gradient norm, for logging and clipping. */
  gradNorm() {
    let sum = 0;
    for (const g of this.grads) {
      for (let i = 0; i < g.length; i++) sum += g[i] * g[i];
    }
    return Math.sqrt(sum);
  }

  /** Scale all gradients by `factor` (used for global norm clipping). */
  scaleGrads(factor) {
    for (const g of this.grads) {
      for (let i = 0; i < g.length; i++) g[i] *= factor;
    }
  }

  clipGradNorm(maxNorm) {
    const norm = this.gradNorm();
    if (norm > maxNorm && norm > 0) this.scaleGrads(maxNorm / norm);
    return norm;
  }

  step() {
    this.t++;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k];
      const g = this.grads[k];
      const m = this.m[k];
      const v = this.v[k];
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] + this.weightDecay * p[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * gi;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * gi * gi;
        p[i] -= (this.lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + this.eps);
      }
    }
  }
}

/** Local mulberry32 copy so net.js has no import cycle with math.js consumers. */
function mulberry(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export { mulberry as mulberryRng, randInt };
