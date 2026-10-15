import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "humanoid_latest.pkl"

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"No checkpoint found at {CHECKPOINT_PATH}. Train the humanoid first with train_walking.py."
    )

env = envs.create("humanoid")
mj_model = env.sys.mj_model
mj_data = mujoco.MjData(mj_model)
params = brax_model.load_params(str(CHECKPOINT_PATH))

network = ppo_networks.make_ppo_networks(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
)
make_policy = ppo_networks.make_inference_fn(network)
policy = make_policy(params, deterministic=True)

jit_policy_fn = jax.jit(policy)
jit_step_fn = jax.jit(env.step)
jit_reset_fn = jax.jit(env.reset)

rng = jax.random.PRNGKey(0)
rng, env_rng = jax.random.split(rng)
state = jit_reset_fn(env_rng)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    print("Viewer started. Press space in the MuJoCo window to pause or resume.")
    while viewer.is_running():
        start = time.time()
        rng, step_rng = jax.random.split(rng)
        action, _ = jit_policy_fn(state.obs, step_rng)
        state = jit_step_fn(state, action)
        mj_data.qpos = np.asarray(jax.device_get(state.pipeline_state.q))
        mj_data.qvel = np.asarray(jax.device_get(state.pipeline_state.qd))
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()
        sleep_for = mj_model.opt.timestep - (time.time() - start)
        if sleep_for > 0:
            time.sleep(sleep_for)