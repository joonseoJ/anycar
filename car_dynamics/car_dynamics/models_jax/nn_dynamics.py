from copy import deepcopy
import os
import jax
import jax.numpy as jnp
import time
import orbax
import optax
import numpy as np
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_foundation.jax_models import JaxDynamicsPredictor
from termcolor import colored
from flax.training import orbax_utils, train_state
from functools import partial
import torch

from car_foundation.torch_models import DynamicsPredictor
from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar

def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = jnp.arctan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    return d_yaw_aligned + yaw_2

class DynamicsJax:
    
    def __init__(self, params: dict):
        self.params = deepcopy(params)
        
        self.key = jax.random.PRNGKey(123)

        self.load_jax_model()

        # # Print the model structure, and count the number of parameters
        # param_count = sum(x.size for x in jax.tree_leaves(raw_restored['model']['params']))
        # print(f"Number of parameters: {param_count}")

        # self.var['params'] = raw_restored['model']['params']
        # self.input_mean = jnp.array(raw_restored['input_mean'])
        # self.input_std = jnp.array(raw_restored['input_std'])
        
        # print("input_mean", self.input_mean)
        # print("input_std", self.input_std)

    def load_jax_model(self):
        self.model = model = JaxDynamicsPredictor(
            model_dim=self.params['model_dim'],
            output_dim=self.params['state_dim']
        )
        dummy_hist = jnp.ones((1, self.params['history_length'], self.params['num_entities'], self.params['history_dim']))
        dummy_static = jnp.ones((1, self.params['num_entities'], self.params['static_dim']))

        self.key, key2 = jax.random.split(self.key, 2)
        self.var = self.model.init(key2, dummy_hist, dummy_static)
        params = self.var['params']

        from car_foundation.train_jax_dynamics_predictor import TrainState
        self.model_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adamw(1e-4),
            rng=self.key
        )

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        manager = orbax.checkpoint.CheckpointManager(
            self.params['model_path'],
            checkpointer,
            orbax.checkpoint.CheckpointManagerOptions()
        )

        step = manager.latest_step()
        print("Restoring checkpoint at step:", step)

        restore_target = {
            'model': self.model_state,
            'config': {
                'target_scale': 100.0,
                'dims': (self.params['history_dim'], self.params['static_dim'])
            }
        }

        self.ckpt = manager.restore(
            step,
            restore_target
        )    
        
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, history: jax.Array, state: jax.Array, action: jax.Array, static_features: jax.Array):
        """
        
        History: (Batch, L, E, H)
        State: (Batch, E, X)
        Action: (Batch, T_future, E, A)
        """
        st_nn_dyn = time.time()

        action_seq = jnp.swapaxes(action, 0, 1) # (T_future, Batch, E, A)

        def scan_step(carry, action_t):
            key, history, state = carry

            # Update history
            current_history = jnp.concatenate(
                [state, action_t], axis=-1
            )  # (Batch, E, H)

            history = jnp.concatenate(
                [history[:, 1:], current_history[:, None]],
                axis=1
            )

            key, subkey = jax.random.split(key)

            pred_delta = self.model_state.apply_fn(
                {'params': self.model_state.params},
                history,
                static_features,
                rngs=subkey,
            ) / self.ckpt['config']['target_scale']

            state = state + pred_delta

            return (key, history, state), None

        (key, history, state), _ = jax.lax.scan(
            scan_step,
            (key, history, state),
            action_seq,
        )
            
        print("NN Inference Time", time.time() - st_nn_dyn)
        return history[:,-action.shape[1]:, :,:self.params['state_dim']]
        