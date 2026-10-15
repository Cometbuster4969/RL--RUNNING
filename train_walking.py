"""Train a humanoid to walk with Brax PPO, warm-starting from an existing checkpoint.

Logs evidence-based locomotion metrics (forward speed, episode length) instead of
reward thresholds, since the brax humanoid alive bonus otherwise dominates reward.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
from brax import envs
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.agents.ppo.train import train as ppo_train

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "training_log.md"
CHECKPOINT_DIR = ROOT / "checkpoints"
LATEST_MODEL_PATH = CHECKPOINT_DIR / "humanoid_latest.pkl"
PREV_MODEL_PATH = CHECKPOINT_DIR / "humanoid_previous.pkl"
STEPS_PATH = CHECKPOINT_DIR / "humanoid_steps.txt"

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices=None):
        if devices is None:
            devices = jax.local_devices()
        return jax.pmap(lambda _: x, devices=devices)(jax.numpy.arange(len(devices)))

    jax.device_put_replicated = _device_put_replicated

ENV_NAME = "humanoid"
NUM_TIMESTEPS = 5_000_000
NUM_ENVS = 16
ACTION_REPEAT = 1
EPISODE_LENGTH = 1000
DISCOUNTING = 0.97
GAE_LAMBDA = 0.95
UNROLL_LENGTH = 10
BATCH_SIZE = 32
NUM_MINIBATCHES = 8
NUM_UPDATES_PER_BATCH = 4
LEARNING_RATE = 2.5e-4
ENTROPY_COST = 1e-3
REWARD_SCALING = 0.1
CLIPPING_EPSILON = 0.2
MAX_GRAD_NORM = 0.9
NORMALIZE_OBSERVATIONS = True
SEED = 0
NUM_EVALS = 20
NUM_EVAL_ENVS = 4
FORWARD_REWARD_WEIGHT = 2.5
HEALTHY_REWARD = 1.0
CTRL_COST_WEIGHT = 0.1
SIM_DT = 0.015


def log_event(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(message, flush=True)


def build_env() -> Any:
    return envs.create(
        env_name=ENV_NAME,
        episode_length=EPISODE_LENGTH,
        forward_reward_weight=FORWARD_REWARD_WEIGHT,
        healthy_reward=HEALTHY_REWARD,
        ctrl_cost_weight=CTRL_COST_WEIGHT,
    )


def metric_value(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def true_episode_steps(metrics: Mapping[str, Any], reported_steps: float) -> float:
    """Recover the real step count from eval metrics.

    brax reports eval/avg_episode_length from info['steps'], which advances twice
    per env step under envs.training.wrap, while episode metric sums advance once.
    Dividing by the reported count therefore halves every per-step rate. The
    alive bonus is exactly HEALTHY_REWARD on every surviving step, so its sum
    divided by HEALTHY_REWARD recovers the true count without a hardcoded factor.
    """
    alive = metric_value(metrics, "eval/episode_reward_alive")
    if HEALTHY_REWARD <= 0.0 or alive != alive or alive <= 0.0:
        return reported_steps
    return alive / HEALTHY_REWARD


def classify_stage(forward_speed: float, episode_length: float) -> str:
    if forward_speed != forward_speed or episode_length != episode_length:
        return "unknown"
    if forward_speed > 1.0 and episode_length > 500:
        return "gait_sustained"
    if forward_speed > 0.5:
        return "locomotion_emerging"
    if episode_length > 400:
        return "upright_but_standing"
    if episode_length > 150:
        return "balance_holding"
    return "falling_early"


def progress_fn(num_steps: int, metrics: Mapping[str, Any]) -> None:
    reported_length = metric_value(metrics, "eval/avg_episode_length")
    steps = true_episode_steps(metrics, reported_length)
    x_velocity = metric_value(metrics, "eval/episode_x_velocity")
    reward = metric_value(metrics, "eval/episode_reward")
    distance = metric_value(metrics, "eval/episode_x_position")
    sps = metric_value(metrics, "training/sps")
    divisor = max(steps, 1.0)
    forward_speed = x_velocity / divisor
    x_travel = forward_speed * steps * SIM_DT
    message = (
        f"| steps: {num_steps:<9d} | reward: {reward:<8.1f} | up_steps: {steps:<6.1f} | "
        f"fwd_speed: {forward_speed:<7.3f} m/s | x_travel: {x_travel:<6.2f} m | "
        f"x_pos: {distance:<7.2f} | sps: {sps:<7.1f} | "
        f"stage: {classify_stage(forward_speed, steps)}"
    )
    log_event(message)


def rollout(policy: Any, env: Any, seed: int, steps: int) -> dict[str, float]:
    """Deterministic episode under a single compiled scan.

    A Python loop over env.step compiles and allocates per call, which exhausts
    the page file on CPU. lax.scan keeps it to one XLA program and masks the
    steps after termination instead of stopping early.
    """
    rng = jax.random.PRNGKey(seed)

    @jax.jit
    def run(rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state = env.reset(rng)

        def body(carry, _):
            state, rng, alive = carry
            rng, step_key = jax.random.split(rng)
            action, _ = policy(state.obs, step_key)
            state = env.step(state, action)
            alive = alive * (1.0 - state.done)
            return (state, rng, alive), (state.reward * alive, state.metrics["x_velocity"] * alive, alive)

        init = (state, rng, jax.numpy.ones(()))
        (_, _, _), (reward, x_velocity, alive) = jax.lax.scan(body, init, (), length=steps)
        return reward.sum(), x_velocity.sum(), alive.sum()

    total_reward, total_x_velocity, length = run(rng)
    length_f = float(length)
    divisor = max(length_f, 1.0)
    forward_speed = float(total_x_velocity) / divisor
    return {
        "reward": float(total_reward),
        "up_steps": length_f,
        "fwd_speed": forward_speed,
        "x_travel": forward_speed * length_f * SIM_DT,
    }


def save_checkpoint(params: Any, cumulative_steps: int) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if LATEST_MODEL_PATH.exists():
        PREV_MODEL_PATH.write_bytes(LATEST_MODEL_PATH.read_bytes())
    model.save_params(str(LATEST_MODEL_PATH), params)
    STEPS_PATH.write_text(f"{cumulative_steps}\n", encoding="utf-8")


def read_cumulative_steps() -> int:
    try:
        return int(STEPS_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=NUM_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--num-evals", type=int, default=NUM_EVALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from checkpoints/humanoid_latest.pkl when it exists",
    )
    parser.add_argument(
        "--reset-critic",
        action="store_true",
        help="keep a freshly initialized value network instead of restoring the old one",
    )
    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=ENTROPY_COST,
        help="entropy bonus weight; lower values sharpen the action distribution "
        "so sampled playback matches the deterministic gait",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "adaptive_kl"],
        default="adaptive_kl",
        help="adaptive_kl anneals the LR when updates move the policy too far, "
        "which settles a gait that has stopped improving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = build_env()
    eval_env = build_env()
    net_ref: dict[str, Any] = {}
    latest: dict[str, Any] = {}

    def make_networks(
        obs_size: int, action_size: int, preprocess_observations_fn: Any
    ) -> ppo_networks.PPONetworks:
        nets = ppo_networks.make_ppo_networks(
            obs_size, action_size, preprocess_observations_fn
        )
        net_ref["networks"] = nets
        return nets

    restore_params = None
    warm_start_steps = read_cumulative_steps()
    if args.warm_start and LATEST_MODEL_PATH.exists():
        restore_params = model.load_params(str(LATEST_MODEL_PATH))
        log_event(
            f"warm_start | path={LATEST_MODEL_PATH} | recorded_prior_steps={warm_start_steps} "
            f"| critic={'fresh' if args.reset_critic else 'restored'}"
        )
    else:
        log_event("cold_start | no checkpoint reused")

    log_event(
        f"session_started | env=brax_{ENV_NAME} (non-MJX, deprecated) backend=cpu "
        f"| timesteps={args.timesteps} num_envs={args.num_envs} evals={args.num_evals}"
    )
    log_event(
        f"reward_shaping | forward_reward_weight={FORWARD_REWARD_WEIGHT} "
        f"healthy_reward={HEALTHY_REWARD} ctrl_cost_weight={CTRL_COST_WEIGHT} "
        f"| rationale=alive bonus cut so locomotion pays more than standing still"
    )
    log_event(
        f"optimizer | learning_rate={LEARNING_RATE} schedule={args.lr_schedule} desired_kl=0.01 "
        f"entropy_cost={args.entropy_cost} | resume_from_step={warm_start_steps}"
    )
    log_event(
        "tracking | fwd_speed = mean(eval/episode_x_velocity) in m/s; "
        "up_steps recovered from eval/episode_reward_alive because brax reports "
        "eval/avg_episode_length at 2x the real step count"
    )

    def policy_params_fn(num_steps: int, make_policy: Any, params: Any) -> None:
        latest["params"] = params
        latest["steps"] = int(num_steps)
        try:
            save_checkpoint(params, warm_start_steps + int(num_steps))
        except Exception as exc:
            log_event(f"checkpoint_write_failed | step={num_steps} | {type(exc).__name__}: {exc}")

    start = time.time()
    interrupted = False
    try:
        _, params, _ = ppo_train(
            environment=env,
            eval_env=eval_env,
            num_timesteps=args.timesteps,
            num_envs=args.num_envs,
            action_repeat=ACTION_REPEAT,
            episode_length=EPISODE_LENGTH,
            discounting=DISCOUNTING,
            gae_lambda=GAE_LAMBDA,
            unroll_length=UNROLL_LENGTH,
            batch_size=BATCH_SIZE,
            num_minibatches=NUM_MINIBATCHES,
            num_updates_per_batch=NUM_UPDATES_PER_BATCH,
        learning_rate=LEARNING_RATE,
        entropy_cost=args.entropy_cost,
        reward_scaling=REWARD_SCALING,
        clipping_epsilon=CLIPPING_EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_observations=NORMALIZE_OBSERVATIONS,
        seed=args.seed,
        num_evals=args.num_evals,
        num_eval_envs=NUM_EVAL_ENVS,
        network_factory=make_networks,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        restore_params=restore_params,
        restore_value_fn=not args.reset_critic,
        learning_rate_schedule=ppo_optimizer.LRSchedule[args.lr_schedule.upper()],
        desired_kl=0.01,
        learning_rate_schedule_min_lr=1e-5,
        learning_rate_schedule_max_lr=2.5e-4,
    )
    except KeyboardInterrupt:
        interrupted = True
        log_event("interrupted | KeyboardInterrupt, saving latest available params")
        params = latest.get("params")
    except Exception as exc:
        log_event(f"training_failed | type={type(exc).__name__} | message={exc}")
        raise

    elapsed = time.time() - start
    session_steps = int(latest.get("steps", 0))
    cumulative = warm_start_steps + session_steps
    log_event(
        f"session_finished | elapsed_seconds={elapsed:.1f} interrupted={interrupted} "
        f"| session_steps={session_steps} cumulative_steps={cumulative} "
        f"| avg_sps={session_steps / elapsed if elapsed > 0 else 0:.1f}"
    )

    if params is None:
        log_event("no_params_available | nothing to checkpoint")
        return

    save_checkpoint(params, cumulative)
    log_event(f"checkpoint_saved | path={LATEST_MODEL_PATH} cumulative_steps={cumulative}")

    try:
        loaded = model.load_params(str(LATEST_MODEL_PATH))
        make_policy = ppo_networks.make_inference_fn(net_ref["networks"])
        policy = make_policy(loaded, deterministic=True)
        stats = rollout(policy, eval_env, seed=args.seed + 1, steps=EPISODE_LENGTH)
        log_event(
            f"deterministic_rollout | up_steps={stats['up_steps']:.0f} "
            f"fwd_speed={stats['fwd_speed']:.3f} m/s x_travel={stats['x_travel']:.2f} m "
            f"reward={stats['reward']:.1f} | stage={classify_stage(stats['fwd_speed'], stats['up_steps'])}"
        )
        log_event(
            f"verdict | {classify_stage(stats['fwd_speed'], stats['up_steps'])} "
            f"at {stats['fwd_speed']:.3f} m/s over {stats['up_steps']:.0f} upright steps"
        )
    except Exception as exc:
        log_event(f"rollout_validation_failed | type={type(exc).__name__} | message={exc}")


if __name__ == "__main__":
    main()
