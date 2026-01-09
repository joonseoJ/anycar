import jax.numpy as jnp

sigma = 0.05
H=100
a_mean = jnp.zeros((H, 2))
sigmas = jnp.array([sigma] * 2)
a_cov_per_step = jnp.diag(sigmas ** 2)
a_cov = jnp.tile(a_cov_per_step[None, :, :], (H, 1, 1))
a_mean_init = a_mean[-1:]
a_cov_init = a_cov[-1:]


import os
import torch
from car_foundation.car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR, CAR_FOUNDATION_LOG_DIR

from car_foundation.torch_models import DynamicsPredictor
from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar

tmp_env = MuJoCoCar({'is_render': False})
model = DynamicsPredictor(
    observation_space=tmp_env.observation_space,
    model_dim=128
).to('cuda')
tmp_env.close()

model_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, '2026-01-07T14:11:07.903-model_checkpoint')
state_dict = torch.load(model_path, map_location="cuda")
model.load_state_dict(state_dict)
model.eval()

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training import train_state
from orbax.checkpoint import utils as orbax_utils
from car_foundation.jax_models import JaxDynamicsPredictor

class TrainState(train_state.TrainState):
    rng: jax.Array

model = JaxDynamicsPredictor(
    model_dim=128,
    output_dim=13
)

# 예시 차원 (저장 당시와 동일해야 함)
BATCH = 1
T = 100
NUM_ENTITIES = 5
HISTORY_DIM = 19
STATIC_DIM = 6

dummy_hist = jnp.ones((BATCH, T, NUM_ENTITIES, HISTORY_DIM))
dummy_static = jnp.ones((BATCH, NUM_ENTITIES, STATIC_DIM))

rng = jax.random.PRNGKey(0)
variables = model.init(rng, dummy_hist, dummy_static)
params = variables['params']

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(1e-4),
    rng=rng
)

MODEL_SAVE_PATH = os.path.join(CAR_FOUNDATION_MODEL_DIR, '2026-01-08T15:48:33.113-model_checkpoint')

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
manager = orbax.checkpoint.CheckpointManager(
    MODEL_SAVE_PATH,
    checkpointer,
    orbax.checkpoint.CheckpointManagerOptions()
)

step = manager.latest_step()
print("Restoring checkpoint at step:", step)

restore_target = {
    'model': state,
    'config': {
        'target_scale': 100.0,
        'dims': (HISTORY_DIM, STATIC_DIM)
    }
}

ckpt = manager.restore(
    step,
    restore_target
)

hist_jax = jnp.ones((1, 10, NUM_ENTITIES, HISTORY_DIM))
static_jax = jnp.ones((1, NUM_ENTITIES, STATIC_DIM))

key = jax.random.PRNGKey(123)
key, key2 = jax.random.split(key, 2)

pred = state.apply_fn(
    {'params': state.params},
    hist_jax,
    static_jax,
    rngs = key2,
)
