import unittest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.struct
import os

# 이전에 작성한 MPPIController와 관련 클래스들이 mppi_jax.py에 있다고 가정합니다.
# 실제 파일명에 맞춰 import 경로를 수정해주세요.
from car_dynamics.controllers_jax.mppi import MPPIController, MPPIParams, MPPIRunningParams
from car_dynamics.controllers_jax.mppi_helper import rollout_fn_jax
from car_dynamics.models_jax import DynamicsJax
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_ros2.utils import load_mppi_params, load_dynamic_params


resume_model_name = "2026-01-08T15:48:33.113-model_checkpoint"
resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name)
dynamics = DynamicsJax({
    'model_path': resume_model_folder_path,
    'model_dim': 128,
    'state_dim': 13,
    'action_dim': 6,
    'static_dim': 6,
    'history_dim': 19,
    'history_length': 100,
    'num_entities': 5,
})
mppi_rollout_fn = rollout_fn_jax(dynamics)

def mock_rollout_start_fn():
    pass

# --- Test Case Class ---
class TestMPPIController(unittest.TestCase):
    def setUp(self):
        # 1. Hyperparameters
        self.N = 12   # Rollouts
        self.T = 100    # History Length
        self.E = 5    # Entities
        self.X = 13    # State Dim
        self.A = 6    # Action Dim
        self.S = 6 #9    # Static Feature Dim
        self.H_dim = self.X + self.A # History Feature Dim
        
        # Horizon Length Calculation
        self.h_knot = 5
        self.num_intermediate = 2
        self.Horizon = (self.h_knot - 1) * self.num_intermediate + 1

        self.params = load_mppi_params()
        
        self.key = jax.random.PRNGKey(0)

        # 2. Initialize Controller
        self.controller = MPPIController(
            self.params, 
            rollout_fn=mppi_rollout_fn, 
            rollout_start_fn=mock_rollout_start_fn, 
            key=self.key
        )

        # 3. Setup Dummy Data
        self.dummy_history = jnp.zeros((self.T, self.E, self.H_dim))
        self.dummy_static = jnp.zeros((self.E, self.S))
        self.dummy_current_state = jnp.zeros((self.E, self.X))
        
        # Initialize Running Params
        self.running_params = self.controller.get_init_params()
        # Overwrite state_hist with correct shape for multi-entity
        self.running_params = self.running_params.replace(
            state_hist = self.dummy_history,
            key = self.key
        )

    def test_init_shapes(self):
        """Test if buffers are initialized with correct shapes."""
        print("\n=== Test Init Shapes ===")
        print(f"Horizon (H): {self.controller.H}")
        self.assertEqual(self.controller.a_mean_flattened.shape, (self.controller.H, self.E * self.A))
        self.assertEqual(self.controller.a_cov_flatten.shape, (self.controller.H, self.E*self.A, self.E*self.A))
        print("Init shapes verified.")


    def test_get_rollout_nn(self):
        """
        Test the full rollout loop (_get_rollout_nn).
        Verifies tiling and scan loop output dimensions.
        """
        print("\n=== Test _get_rollout_nn (Full Trajectory) ===")
        
        # Inputs
        # actions: (N, Horizon, A)
        actions = jnp.zeros((self.N, self.Horizon, self.E, self.A))
        key = jax.random.PRNGKey(2)
        
        state_list = self.controller._get_rollout(
            key,
            self.dummy_current_state, # (1, E, X)
            self.dummy_history,       # (1, T, E, H_dim)
            actions,                  # (N, Horizon, E, A)
            self.dummy_static,        # (1, E, S)
            fix_history=False
        )
        
        # Expected Output: (N, Horizon, E, X)
        expected_shape = (self.Horizon+1, self.N, self.E, self.X)
        self.assertEqual(state_list.shape, expected_shape, f"State list shape mismatch. Got {state_list.shape}, Expected {expected_shape}")
        print("Full rollout dimensions verified.")

    def test_integration_call(self):
        """
        Test the main __call__ function.
        Verifies the end-to-end MPPI loop.
        """
        print("\n=== Test __call__ (Integration) ===")
        
        goal_list = jnp.zeros((self.controller.H+1, 4)) # Dummy goals
        
        # Run Control Step
        u, new_params, info = self.controller(
            obs=self.dummy_current_state,
            goal_list=goal_list,
            running_params=self.running_params,
            static_features=self.dummy_static,
        )
        
        # 1. Output Action Shape
        self.assertEqual(u.shape, (self.E, self.A,), "Optimal action u shape mismatch")
        
        # 2. Updated Params Shape
        self.assertEqual(new_params.a_mean_flattened.shape, (self.controller.H, self.E*self.A), "Updated a_mean shape mismatch")
        self.assertEqual(new_params.a_cov_flattened.shape, (self.controller.H, self.E*self.A, self.E*self.A), "Updated a_cov shape mismatch")
        
        # 3. Trajectory Info Shape
        # optim_traj: (Horizon, E, X) - from single rollout of mean action
        optim_traj = info['trajectory']
        self.assertEqual(optim_traj.shape, (self.controller.H+1, self.E, self.X), "Optimized trajectory shape mismatch")
        
        print(f"Optimal Action: {u}")
        print("Integration test passed.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)