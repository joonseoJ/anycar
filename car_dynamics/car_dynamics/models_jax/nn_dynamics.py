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
            state_dim=self.params['state_dim'],
            num_heads=self.params['num_heads'],
            num_layers=self.params['num_layers']
        )
        dummy_hist = jnp.ones((1, self.params['history_length'], self.params['num_entities'], self.params['history_dim']))
        dummy_static = jnp.ones((1, self.params['num_entities'], self.params['static_dim']))
        dummy_pred_input = jnp.ones((1, 1, self.params['num_entities'], self.params['action_dim']))


        self.key, key2 = jax.random.split(self.key, 2)
        self.var = self.model.init(key2, dummy_hist, dummy_static, dummy_pred_input)
        params = self.var['params']

        self.model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adamw(1e-4)
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
            'description': 'Inference future states (B,T,E,X) without normalizing'
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
        key, key2 = jax.random.split(key, 2)

        y_pred = self.model_state.apply_fn(
            {'params': self.model_state.params},
            history,
            static_features,
            action,
            rngs=key2,
            deterministic=True
        )
            
        print("NN Inference Time", time.time() - st_nn_dyn)
        return y_pred
        