/**
 * Proximal Policy Optimisation, written from scratch.
 *
 * The update is the standard one: collect a rollout, estimate advantages with
 * GAE(lambda), then take several epochs over shuffled minibatches optimising
 *
 *     L = -E[min(r*A, clip(r, 1-e, 1+e)*A)] + c_v * E[(V - R)^2] - c_h * E[H]
 *
 * with r = exp(log pi_new(a|s) - log pi_old(a|s)). The gradient of the clipped
 * surrogate w.r.t. the Gaussian log-probability is written out by hand, and
 * `test/net.test.js` gradient-checks the whole thing against finite differences.
 */
import { Policy, Adam } from './net.js';
import { RunningEnv, OBS_SIZE, ACTION_SIZE, CONTROL_DT } from './env.js';
import { mulberry32 } from './math.js';

export const DEFAULT_HYPER = {
  rolloutSteps: 1024,
  epochs: 3,
  minibatch: 128,
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  lr: 3e-4,
  valueCoef: 0.5,
  entropyCoef: 0.0,
  maxGradNorm: 0.5,
  rewardScale: 0.1,
  targetKL: 0.04,
  weightDecay: 0,
};

/**
 * Trains one agent in one environment. One trainer == one "lane" in the race.
 */
export class PPOTrainer {
  /**
   * @param {{seed?:number, hyper?:Partial<typeof DEFAULT_HYPER>, hidden?:number[],
   *          policy?:Policy, envOpts?:object, env?:RunningEnv, logStdInit?:number}} opts
   */
  constructor(opts = {}) {
    this.seed = opts.seed ?? 1;
    this.hyper = { ...DEFAULT_HYPER, ...(opts.hyper || {}) };
    this.rng = mulberry32(this.seed * 2654435761 + 12345);
    this.actionRng = mulberry32(this.seed * 40503 + 7);

    this.env = opts.env || new RunningEnv({ seed: this.seed * 977 + 3, randomize: true, ...(opts.envOpts || {}) });
    this.policy =
      opts.policy ||
      new Policy({
        obsSize: OBS_SIZE,
        actSize: ACTION_SIZE,
        hidden: opts.hidden ?? [64, 64],
        seed: this.seed,
        logStdInit: opts.logStdInit ?? -0.7,
      });

    // Dimensions always come from the policy itself: hard-coding them would make
    // the trainer silently mis-index any network whose sizes differ.
    this.obsSize = this.policy.obsSize;
    this.actSize = this.policy.actSize;
    if (this.env.obs.length !== this.obsSize) {
      throw new Error(`policy expects ${this.obsSize} observations but the env provides ${this.env.obs.length}`);
    }
    if (ACTION_SIZE !== this.actSize) {
      throw new Error(`policy expects ${this.actSize} actions but the env provides ${ACTION_SIZE}`);
    }

    const { params, grads } = this.policy.collect();
    this.adam = new Adam(params, grads, { lr: this.hyper.lr, weightDecay: this.hyper.weightDecay });

    this.obsBuf = new Float64Array(this.hyper.rolloutSteps * this.obsSize);
    this.actBuf = new Float64Array(this.hyper.rolloutSteps * this.actSize);
    this.logpBuf = new Float64Array(this.hyper.rolloutSteps);
    this.valBuf = new Float64Array(this.hyper.rolloutSteps);
    this.rewBuf = new Float64Array(this.hyper.rolloutSteps);
    this.doneBuf = new Uint8Array(this.hyper.rolloutSteps);
    this.advBuf = new Float64Array(this.hyper.rolloutSteps);
    this.retBuf = new Float64Array(this.hyper.rolloutSteps);
    this.order = new Int32Array(this.hyper.rolloutSteps);
    // Reused minibatch gather buffers (sized for the largest possible minibatch).
    const MB = this.hyper.minibatch;
    this.mbObs = new Float64Array(MB * this.obsSize);
    this.mbAct = new Float64Array(MB * this.actSize);
    this.mbAdv = new Float64Array(MB);
    this.mbLogp = new Float64Array(MB);
    this.mbRet = new Float64Array(MB);
    for (let i = 0; i < this.order.length; i++) this.order[i] = i;

    this.steps = 0;
    this.updates = 0;
    this.episodes = 0;
    this.lastUpdateMs = 0;
    this.kl = 0;
    this.gradNorm = 0;
    this.valueLoss = 0;
    this.policyLoss = 0;
    this.entropy = 0;
    /**
     * Episode outcomes for plotting.
     *
     * When this grows past `historyCap`, alternate entries are dropped and the
     * resolution halves - so the curve always spans the *entire* run instead of
     * keeping only recent episodes (which would silently lose the interesting
     * part of the learning curve), while memory stays bounded.
     * @type {{step:number, distance:number, reward:number, time:number}[]}
     */
    this.history = [];
    this.historyCap = opts.historyCap ?? 2000;
    this.distanceEma = 0;
    this.windowDistances = [];
    this.bestDistance = 0;
    this.bestReward = -Infinity;
    this.lastObs = Float64Array.from(this.env.obs);
    this.totalEnvSteps = 0;
  }

  /** Collect `rolloutSteps` environment interactions using the current policy. */
  collectRollout() {
    const { policy, env, hyper } = this;
    const T = hyper.rolloutSteps;
    let i = 0;
    while (i < T) {
      const { action, logProb, value } = policy.act(this.lastObs, this.actionRng);
      const obsAtStep = i * this.obsSize;
      for (let k = 0; k < this.obsSize; k++) this.obsBuf[obsAtStep + k] = this.lastObs[k];
      for (let k = 0; k < this.actSize; k++) this.actBuf[i * this.actSize + k] = action[k];
      this.logpBuf[i] = logProb;
      this.valBuf[i] = value;

      const { reward, terminated, truncated, obs } = env.step(action);
      this.rewBuf[i] = reward;
      this.doneBuf[i] = terminated || truncated ? 1 : 0;
      this.lastObs.set(obs);
      i++;
      this.steps++;
      this.totalEnvSteps++;

      if (terminated || truncated) {
        this.episodes++;
        const distance = env.lastDistance;
        this.distanceEma = this.distanceEma === 0 ? distance : this.distanceEma * 0.98 + distance * 0.02;
        this.bestDistance = Math.max(this.bestDistance, distance);
        this.bestReward = Math.max(this.bestReward, env.lastEpisodeReward);
        this._recordEpisode({
          step: this.steps,
          distance,
          reward: env.lastEpisodeReward,
          time: env.lastEpisodeTime,
        });
      }
    }
    // Value of the state that follows the rollout, used to bootstrap the last GAE step.
    const lastValue = policy.forward(this.lastObs).value;
    this.bootstrapValue = lastValue;
    return T;
  }

  /**
   * Append an episode outcome. When the cap is reached the array is thinned by
   * keeping every other entry (always preserving the very first and the very
   * latest), so the curve keeps spanning the whole run with the resolution
   * decaying geometrically towards the start - which is exactly how you want a
   * long learning curve drawn.
   */
  _recordEpisode(entry) {
    this.history.push(entry);
    if (this.history.length <= this.historyCap) return;
    const kept = [];
    for (let i = 0; i < this.history.length; i += 2) kept.push(this.history[i]);
    const last = this.history[this.history.length - 1];
    if (kept[kept.length - 1] !== last) kept.push(last);
    this.history = kept;
  }

  /** Median spacing between history entries, in training steps. */
  historyStride() {
    if (this.history.length < 3) return 1;
    const gaps = [];
    for (let i = 1; i < this.history.length; i++) gaps.push(this.history[i].step - this.history[i - 1].step);
    gaps.sort((a, b) => a - b);
    return gaps[Math.floor(gaps.length / 2)];
  }

  /** GAE(lambda) advantage estimation over the current rollout. */
  computeAdvantages() {
    const { gamma, lambda, rewardScale } = this.hyper;
    const T = this.hyper.rolloutSteps;
    let gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const r = this.rewBuf[t] * rewardScale;
      const notDone = this.doneBuf[t] ? 0 : 1;
      const nextValue = t === T - 1 ? this.bootstrapValue : this.valBuf[t + 1];
      const delta = r + gamma * nextValue * notDone - this.valBuf[t];
      gae = delta + gamma * lambda * notDone * gae;
      this.advBuf[t] = gae;
      this.retBuf[t] = gae + this.valBuf[t];
    }
    // Normalise advantages (mean 0, std 1) for a well-conditioned policy gradient.
    let mean = 0;
    for (let t = 0; t < T; t++) mean += this.advBuf[t];
    mean /= T;
    let varSum = 0;
    for (let t = 0; t < T; t++) {
      const d = this.advBuf[t] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / T) + 1e-8;
    for (let t = 0; t < T; t++) this.advBuf[t] = (this.advBuf[t] - mean) / std;
  }

  /**
   * One PPO update over the rollout: `epochs` passes of shuffled minibatches.
   * Returns diagnostics.
   */
  update() {
    const { policy, adam, hyper } = this;
    const T = hyper.rolloutSteps;
    const M = hyper.minibatch;
    const A = this.actSize;
    const O = this.obsSize;
    const nBatches = Math.max(1, Math.floor(T / M));

    let klSum = 0;
    let klCount = 0;
    let entropySum = 0;
    let valueLossSum = 0;
    let policyLossSum = 0;
    let gradNormSum = 0;
    let earlyStopped = -1;

    for (let epoch = 0; epoch < hyper.epochs; epoch++) {
      // Fisher-Yates shuffle of the sample order.
      for (let i = T - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        const tmp = this.order[i];
        this.order[i] = this.order[j];
        this.order[j] = tmp;
      }

      for (let b = 0; b < nBatches; b++) {
        const size = Math.min(M, T - b * M);
        if (size <= 0) break;
        const X = this.mbObs.subarray(0, size * O);
        const act = this.mbAct.subarray(0, size * A);
        const adv = this.mbAdv.subarray(0, size);
        const oldLogp = this.mbLogp.subarray(0, size);
        const ret = this.mbRet.subarray(0, size);
        for (let sIdx = 0; sIdx < size; sIdx++) {
          const idx = this.order[b * M + sIdx];
          X.set(this.obsBuf.subarray(idx * O, idx * O + O), sIdx * O);
          act.set(this.actBuf.subarray(idx * A, idx * A + A), sIdx * A);
          adv[sIdx] = this.advBuf[idx];
          oldLogp[sIdx] = this.logpBuf[idx];
          ret[sIdx] = this.retBuf[idx];
        }

        policy.zeroGrad();
        const { policyLoss, valueLoss, entropy, kl, gradNorm } = this._minibatchLossGrad(X, act, adv, oldLogp, ret, size);
        adam.clipGradNorm(hyper.maxGradNorm);
        adam.step();

        klSum += kl;
        klCount += size;
        entropySum += entropy;
        valueLossSum += valueLoss;
        policyLossSum += policyLoss;
        gradNormSum += gradNorm;
      }

      // Trust region: stop the update when the policy has already moved too far.
      const approxKL = klCount > 0 ? klSum / klCount : 0;
      if (approxKL > hyper.targetKL) {
        earlyStopped = epoch;
        break;
      }
    }

    this.updates++;
    this.kl = klCount > 0 ? klSum / klCount : 0;
    this.entropy = entropySum / Math.max(1, nBatches * (earlyStopped >= 0 ? earlyStopped + 1 : hyper.epochs));
    this.valueLoss = valueLossSum / Math.max(1, nBatches * (earlyStopped >= 0 ? earlyStopped + 1 : hyper.epochs));
    this.policyLoss = policyLossSum / Math.max(1, nBatches * (earlyStopped >= 0 ? earlyStopped + 1 : hyper.epochs));
    this.gradNorm = gradNormSum / Math.max(1, nBatches * (earlyStopped >= 0 ? earlyStopped + 1 : hyper.epochs));
    this.earlyStopped = earlyStopped >= 0;
    return { kl: this.kl, entropy: this.entropy, valueLoss: this.valueLoss, policyLoss: this.policyLoss };
  }

  /**
   * Forward + backward for one minibatch, accumulating gradients into the policy
   * and returning the diagnostics. This is the only place the PPO maths lives.
   */
  _minibatchLossGrad(X, act, adv, oldLogp, ret, size) {
    const { policy, hyper } = this;
    const A = this.actSize;
    const O = this.obsSize;
    const M = size;

    // --- forward -----------------------------------------------------------
    let h = X;
    const inputs = [X];
    for (const layer of policy.trunk) {
      h = layer.forward(h, M);
      inputs.push(h);
    }
    const hLast = h;
    const meanOut = policy.meanHead.forward(hLast, M);
    const valueOut = policy.valueHead.forward(hLast, M);

    const std = new Float64Array(A);
    for (let i = 0; i < A; i++) std[i] = Math.exp(Math.min(1, Math.max(-4, policy.logStd[i])));

    // --- loss --------------------------------------------------------------
    const dLogProb = this._dLogProb && this._dLogProb.length >= M * A ? this._dLogProb.subarray(0, M * A) : (this._dLogProb = new Float64Array(M * A));
    const dValue = this._dValue && this._dValue.length >= M ? this._dValue.subarray(0, M) : (this._dValue = new Float64Array(M));
    dLogProb.fill(0);
    dValue.fill(0);
    let policyLoss = 0;
    let valueLoss = 0;
    let entropy = 0;
    let kl = 0;
    let clippedFraction = 0;

    // New log-probability of the sampled actions, and the PPO ratios.
    const newLogp = this._newLogp && this._newLogp.length >= M ? this._newLogp.subarray(0, M) : (this._newLogp = new Float64Array(M));
    for (let s = 0; s < M; s++) {
      let logp = 0;
      for (let i = 0; i < A; i++) {
        const dev = (act[s * A + i] - meanOut[s * A + i]) / std[i];
        logp += -0.5 * dev * dev - Math.log(std[i]) - 0.5 * Math.log(2 * Math.PI);
      }
      newLogp[s] = logp;
    }

    for (let s = 0; s < M; s++) {
      const logp = newLogp[s];
      const logpOld = oldLogp[s];
      const ratio = Math.exp(logp - logpOld);
      const a = adv[s];
      const unclipped = ratio * a;
      const clipped = Math.min(1 + hyper.clip, Math.max(1 - hyper.clip, ratio)) * a;
      const useUnclipped = unclipped <= clipped;
      if (!useUnclipped) clippedFraction++;
      policyLoss += -Math.min(unclipped, clipped) / M;

      // dLoss/dlogpi = -A * ratio / M inside the trust region, 0 outside it.
      const dLogp = useUnclipped ? (-a * ratio) / M : 0;
      for (let i = 0; i < A; i++) {
        const dev = (act[s * A + i] - meanOut[s * A + i]) / std[i];
        // dlogp/dmu = (a - mu) / sigma^2 ; dlogp/dlogsigma = dev^2 - 1
        dLogProb[s * A + i] = dLogp * (dev / std[i]);
        policy.gLogStd[i] += dLogp * (dev * dev - 1);
      }

      const v = valueOut[s];
      dValue[s] = (hyper.valueCoef * (v - ret[s])) / M;
      valueLoss += (0.5 * hyper.valueCoef * (v - ret[s]) * (v - ret[s])) / M;

      let entropyPerSample = 0;
      for (let i = 0; i < A; i++) {
        entropyPerSample += 0.5 * Math.log(2 * Math.PI * Math.E) + Math.log(std[i]);
      }
      entropy += entropyPerSample / M;
      kl += logpOld - logp;
    }

    if (hyper.entropyCoef > 0) {
      // d/dlogstd of the mean entropy is 1 per dimension.
      for (let i = 0; i < A; i++) policy.gLogStd[i] -= hyper.entropyCoef;
    }

    // --- backward ----------------------------------------------------------
    const dH = policy.meanHead.backward(dLogProb, hLast, M);
    const dHValue = policy.valueHead.backward(dValue, hLast, M);
    for (let i = 0; i < dH.length; i++) dH[i] += dHValue[i];
    let dCur = dH;
    for (let i = policy.trunk.length - 1; i >= 0; i--) {
      dCur = policy.trunk[i].backward(dCur, inputs[i], M);
    }

    return {
      policyLoss,
      valueLoss,
      entropy,
      kl,
      gradNorm: this.adam.gradNorm(),
      clippedFraction: clippedFraction / M,
    };
  }

  /** Run `steps` environment steps, updating in rollouts of `rolloutSteps`. */
  trainSteps(steps) {
    let done = 0;
    while (done < steps) {
      const n = Math.min(this.hyper.rolloutSteps, steps - done);
      if (n === this.hyper.rolloutSteps) {
        this.collectRollout();
        this.computeAdvantages();
        this.update();
      } else {
        // Partial rollout: temporarily shrink the rollout length.
        const saved = this.hyper.rolloutSteps;
        this.hyper.rolloutSteps = Math.max(32, n);
        this.collectRollout();
        this.computeAdvantages();
        this.update();
        this.hyper.rolloutSteps = saved;
      }
      done += n;
    }
    return this.steps;
  }

  /**
   * Run `episodes` evaluation episodes with the greedy policy (no exploration
   * noise) and return distance statistics.
   */
  evaluate(episodes = 3, opts = {}) {
    const greedy = opts.greedy ?? true;
    const env = opts.env || new RunningEnv({ seed: this.seed * 31 + 1000, randomize: opts.randomize ?? false });
    void env;
    const results = [];
    const action = new Float64Array(this.actSize);
    for (let e = 0; e < episodes; e++) {
      let done = false;
      let steps = 0;
      while (!done && steps < 60 / CONTROL_DT) {
        if (greedy) {
          action.set(env.obs ? this.policy.actDeterministic(env.obs) : action);
        } else {
          action.set(this.policy.act(Float64Array.from(env.obs), this.actionRng).action);
        }
        const { terminated, truncated } = env.step(action);
        done = terminated || truncated;
        steps++;
      }
      results.push({ distance: env.lastDistance, time: env.lastEpisodeTime, fell: env.terminatedByFall });
    }
    const distances = results.map((r) => r.distance);
    const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
    return {
      mean,
      best: Math.max(...distances),
      fallRate: results.filter((r) => r.fell).length / results.length,
      results,
      speed: mean / Math.max(0.1, results.reduce((a, r) => a + r.time, 0) / results.length),
    };
  }

  /** Lightweight stats for the UI/CLI. */
  stats() {
    const recent = this.history.slice(-20);
    const meanRecent = recent.length ? recent.reduce((a, h) => a + h.distance, 0) / recent.length : 0;
    return {
      steps: this.steps,
      updates: this.updates,
      episodes: this.episodes,
      best: this.bestDistance,
      recent: meanRecent,
      ema: this.distanceEma,
      kl: this.kl,
      entropy: this.entropy,
      valueLoss: this.valueLoss,
      policyLoss: this.policyLoss,
      gradNorm: this.gradNorm,
      history: this.history,
      env: {
        distance: this.env.lastDistance,
        episode: this.env.episode,
        time: this.env.time,
        x: this.env.trunk.x,
      },
    };
  }
}

/** Build a trainer, optionally warm started from a saved policy JSON object. */
export function createLane(opts = {}) {
  let policy = null;
  if (opts.policyJSON) {
    policy = Policy.fromJSON(opts.policyJSON);
    if (opts.policyPerturb) {
      // Small random kick so lanes from the same checkpoint diverge and explore.
      const rng = mulberry32((opts.seed ?? 1) * 7919 + 101);
      const { params } = policy.collect();
      const sigma = opts.policyPerturb;
      for (const p of params) {
        for (let i = 0; i < p.length; i++) p[i] += gaussianish(rng) * sigma * (Math.abs(p[i]) + 0.1);
      }
    }
  }
  return new PPOTrainer({ ...opts, policy });
}

function gaussianish(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
