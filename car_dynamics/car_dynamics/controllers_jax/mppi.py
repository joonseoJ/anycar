from dataclasses import dataclass
import jax
import numpy as np
import jax.numpy as jnp
import time
from termcolor import colored
from .base import BaseController
from functools import partial
from car_dynamics.models_jax import DynamicParams   
import flax
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from car_dynamics.controllers_jax.utils import quat_to_yaw

@dataclass
class MPPIParams:
    sigma: float
    gamma_mean: float
    gamma_sigma: float
    discount: float
    sample_sigma: float
    lam: float
    n_rollouts: int
    h_knot: int
    a_min: jnp.ndarray
    a_max: jnp.ndarray
    a_mag: jnp.ndarray
    a_shift: jnp.ndarray
    delay: int
    len_history: int
    debug: bool
    fix_history: bool
    num_obs: int
    num_actions: int
    num_entities: int
    num_intermediate: int
    spline_order: int
    smooth_alpha: float = 0.8
    dynamics: str = 'dbm'
    dual: bool = False

@flax.struct.dataclass
class MPPIRunningParams:
    a_mean_flattened: jnp.ndarray
    a_cov_flattened: jnp.ndarray  
    prev_a_flattened: jnp.ndarray
    state_hist: jnp.ndarray
    key: jax.random.PRNGKey

def slow_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
    
    
class MPPIController(BaseController):
    def __init__(self,
                params: MPPIParams, rollout_fn: callable, rollout_start_fn: callable, key):
        """ MPPI implemented in Jax """
        assert params.gamma_sigma <= 0.
        self.params = params
        self.rollout_fn = rollout_fn
        self.rollout_start_fn = rollout_start_fn
        self._init_buffers()
        
        if params.dynamics == 'dbm':
            self.scan_fn = jax.lax.scan
            self._get_rollout = self._get_rollout_dbm
        elif params.dynamics == 'transformer-jax':
            self.scan_fn = self.scan_seq_jax
            self._get_rollout = self._get_rollout_nn
        else:
            raise ValueError(f"Unknown dynamics model: {params.dynamics}")
        
        
    def _init_buffers(self, ):
        self.spline_order = self.params.spline_order
        self.H = (self.params.h_knot -1 ) * self.params.num_intermediate + 1 
        
        flattened_action_dim = self.params.num_entities * self.params.num_actions
        self.a_mean_flattened = jnp.zeros((self.H, flattened_action_dim))
        sigmas_flatten = jnp.ones((flattened_action_dim,)) * self.params.sigma
        a_cov_flatten_per_step = jnp.diag(sigmas_flatten ** 2)
        self.a_cov_flatten = jnp.tile(a_cov_flatten_per_step[None, :, :], (self.H, 1, 1))
        self.a_mean_init = self.a_mean_flattened[-1:]
        self.a_cov_flatten_init = self.a_cov_flatten[-1:]
        
        self.step_us = jnp.arange(self.H)
        self.step_nodes = jnp.arange(self.params.h_knot) * (self.params.num_intermediate)
        self.prev_a_flattened = jnp.zeros((self.params.delay, flattened_action_dim))
        # self.step_count = 0
        
        state_hist_len = max(self.params.len_history, 1)
        self.state_hist_init = jnp.zeros((state_hist_len, self.params.num_entities ,self.params.num_obs + self.params.num_actions))
        
        self.action_sampled_flattened = jnp.zeros((self.params.n_rollouts, self.H, flattened_action_dim))
        self.flattened_action_init_buf = jnp.zeros((self.params.n_rollouts, self.H + self.params.delay, flattened_action_dim))
        self.state_init_buf = jnp.ones((self.params.num_entities, self.params.num_obs))
        self.x_all = []
        self.y_all = []
        
        self.node2u_vmap = jax.vmap(self.node2u, in_axes=(0,))
        self.u2node_vmap = jax.vmap(self.u2node, in_axes=(0,))
        
    @partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=self.spline_order)
        us = spline(self.step_us)
        return us
    
    @partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=self.spline_order)
        nodes = spline(self.step_nodes)
        return nodes
    
    def get_init_params(self, ):
        return MPPIRunningParams(
            a_mean_flattened = self.a_mean_flattened,
            a_cov_flattened = self.a_cov_flatten,
            prev_a_flattened = self.prev_a_flattened,
            key = jax.random.PRNGKey(123),
            state_hist = self.state_hist_init,
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def _running_average(self, carry, x):
        prev_x = carry
        new_x = x * self.params.smooth_alpha + prev_x * (1 - self.params.smooth_alpha)
        return new_x, new_x
    
    @partial(jax.jit, static_argnums=(0,))
    def normalize_action(self, a_sampled):
        for d in range(len(self.params.a_min)):
            a_sampled = a_sampled.at[:, :, d].set(jnp.clip(a_sampled[:, :, d], self.params.a_min[d], self.params.a_max[d]) * self.params.a_mag[d] + self.params.a_shift[d])
        return a_sampled
    
    @partial(jax.jit, static_argnums=(0,))
    def _rollout_jit(self, carry, action):
        state, obs_history, dynamic_params_tuple = carry
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action)
        state, _ = self.rollout_fn(obs_history, state, action, dynamic_params_tuple, self.params.debug)
        obs_history = obs_history.at[:-1].set(obs_history[1:])    
        return (state, obs_history, dynamic_params_tuple), state
    
    def scan_seq(self, rollout_fn, init, action_list):
        state, obs_history, dynamic_params_tuple  = init
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action_list[0])
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        action_list = jnp.swapaxes(action_list, 0, 1)
        state_list, debug_info = self.rollout_fn(obs_history, state, action_list, dynamic_params_tuple, self.params.debug)
        return debug_info, state_list
    
    @partial(jax.jit, static_argnums=(0,))
    def scan_seq_jax(self, key, current_state, obs_history, static_features, action_list):
        # current_state (N_rollout, E, X)
        # obs_history (N_rollout, T, E, H)
        # static_features (N_rollout, E, S)
        # action_list (N_rollout, T_future, E, A)
        
        key, key2 = jax.random.split(key, 2)
        state_list, debug_info = self.rollout_fn(key2, obs_history, current_state, action_list, static_features, self.params.debug)
        return debug_info, state_list
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_rollout_dbm(self, key, state_init, state_hist, actions, dynamic_params_tuple, fix_history=False):
        """
        Get rollout for dynamic bicycle model.

        This method prepares the input data, calls the rollout function, and processes
        the output for the dynamics-based model. It handles the initialization of states,
        observation history, and actions, then performs the rollout using the scan function.

        Args:
            key (jax.random.PRNGKey): Random key for JAX operations.
            state_init (jnp.ndarray): Initial state.
            state_hist (jnp.ndarray): History of previous states.
            actions (jnp.ndarray): Sampled actions for rollouts.
            dynamic_params_tuple (tuple): Parameters for the dynamics model.
            fix_history (bool, optional): Whether to fix the history. Defaults to False.

        Returns:
            jnp.ndarray: Array of states for all timesteps in the rollout.
        """
        n_rollouts = actions.shape[0]
        state = jnp.tile(jnp.expand_dims(state_init, 0), (n_rollouts, 1))
        state_list = state[None]
        obs_history = jnp.tile(jnp.expand_dims(state_hist.copy(), 0), (n_rollouts, 1, 1))
        self.rollout_start_fn()
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)
        # For 1-step model
        _, state_list2 = self.scan_fn(self._rollout_jit, (state, obs_history, dynamic_params_tuple), actions)
        state_list = jnp.concatenate((state_list, state_list2), axis=0)
        state_list_jnp = jnp.array(state_list)
        return state_list_jnp
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_rollout_nn(
            self, 
            key: jax.random.PRNGKey, 
            current_state: jnp.ndarray, 
            state_hist: jnp.ndarray, 
            actions: jnp.ndarray, 
            static_features: jnp.ndarray,
        ):
        """
        Get rollout for neural network-based dynamics model.

        This method prepares the input data, calls the rollout function, and processes
        the output for the neural network-based dynamics model.

        Args:
            key (jax.random.PRNGKey): Random key for JAX operations.
            current_state (jnp.ndarray): State at t (E, X)
            state_hist (jnp.ndarray): History of previous states. (T, E, H)
            actions (jnp.ndarray): Sampled actions for rollouts. (N_rollout, T_future, E, A)
            static_features (jnp.ndarray): Parameters for the dynamics model. (E, S)

        Returns:
            jnp.ndarray: Array of states from t to t+T_future in the rollout. (N_rollout, T_future+1, E, A)
        """
        n_rollouts = actions.shape[0] # N
        n_steps = actions.shape[1]    # Horizon Length

        # 1. Tile Initial State
        # current_state: (E, X) -> (N, E, X)
        current_state = jnp.tile(jnp.expand_dims(current_state, 0), (n_rollouts, 1, 1))

        # 2. Tile History
        # state_hist: (T, E, H) -> (N, T, E, H)
        obs_history = jnp.tile(jnp.expand_dims(state_hist, 0), (n_rollouts, 1, 1, 1))
        obs_history = obs_history.at[:, -1, :, self.params.num_obs:].set(actions[:, 0, :, :])

        # 3. Tile Static Features
        # static_features: (E, S) -> (N, E, S)
        static_features_tiled = jnp.tile(jnp.expand_dims(static_features, 0), (n_rollouts, 1, 1))

        #4. Run static_features_tiled
        # state_list2: (T_future, N, E, X)
        _, state_list2 = self.scan_seq_jax(key, current_state, obs_history, static_features_tiled, actions)

        state_list = current_state[None] #(1, N, E, X)
        state_list = jnp.concatenate((state_list, state_list2), axis=0)
        state_list_jnp = jnp.array(state_list)
        state_list_jnp = jnp.swapaxes(state_list_jnp, 0, 1) #(N, T_future+1, E, X)

        return state_list_jnp
    

    
    @partial(jax.jit, static_argnums=(0,))
    def single_step_reward(self, carry, pair):
        """
        Calculate the reward for a single step in the rollout.

        Args:
            carry (tuple): Contains the current step and previous action.
                current step (int): The current timestep in the rollout.
                previous action (jnp.ndarray): The action taken at the previous timestep. (N, E, A)
            pair (tuple): Contains the current state, action, and goal for this step.
                current state (jnp.ndarray): The state at the current timestep. (N, E, X)
                current action (jnp.ndarray): The action taken at the current timestep. (N, E, A)
                goal (jnp.ndarray): The goal state for this timestep. (x, y, yaw, vx)

        Returns:
            tuple: Updated carry (next step and current action) and the calculated reward.
                reward (jnp.ndarray): The reward for this timestep. (N,)
        """
        current_step, prev_action = carry
        current_state, current_action, goal = pair

        dist_pos = jnp.linalg.norm(current_state[:, 0, :2] - goal[:2], axis=1)
        
        curr_yaw = jax.vmap(quat_to_yaw)(current_state[:, 0, 3:7])
        diff_yaw = curr_yaw - goal[2]
        diff_yaw = jnp.arctan2(jnp.sin(diff_yaw), jnp.cos(diff_yaw))
        diff_vel = current_state[:, 0, 7] - goal[3]

        diff_throttle = jnp.linalg.norm(current_action[:, 1:, 4] - prev_action[:, 1:, 4], axis=1)
        diff_steering = jnp.linalg.norm(current_action[:, 1:, 5] - prev_action[:, 1:, 5], axis=1)
        
        reward_pos_err = -dist_pos ** 2
        reward_psi = -diff_yaw ** 2
        reward_vel = -diff_vel ** 2
        reward_throttle = - diff_throttle ** 2
        reward_steering = - diff_steering ** 2
        reward = reward_pos_err*5.0 + reward_psi*5.0 + reward_vel*1. + reward_throttle*1.0 + reward_steering*1.0
        # reward *= (self.params.discount ** step)
        return (current_step + 1, current_action), reward
    
    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state, action, prev_action, goal_list):
        """
        Calculate the total reward for each rollout trajectory.

        Args:
            state (jnp.ndarray): State trajectory from t to t+T_future for all rollouts. (N, 1+T_future, E, X)
            action (jnp.ndarray): Action trajectory from t to t+T_future-1 for all rollouts. (N_rollout, T_future, E, A)
            prev_action (jnp.ndarray): Action at t-1. (E, A)
            goal_list (jnp.ndarray): List of goal states for t+1 to t+T_future. (T_future, 4)

        Returns:
            jnp.ndarray: Total rewards for each rollout.
        """

        # (N, 1+T_future, E, X) -> (1+T_future, N, E, X)
        state = jnp.swapaxes(state, 0, 1)

        # (N, T_future, E, A) -> (T_future, N, E, A)
        actions = jnp.swapaxes(action, 0, 1)

        # (E, A) -> (N, E, A)
        prev_action = jnp.tile(jnp.expand_dims(prev_action, 0), (actions.shape[1], 1, 1))

        _, reward_list = jax.lax.scan(self.single_step_reward, (0, prev_action), (state[1:], actions, goal_list))
        rewards = jnp.sum(reward_list, axis=0)
        return rewards
    
    
    @partial(jax.jit, static_argnums=(0,)) 
    def feed_hist(self, param: MPPIRunningParams, current_state, prev_action):
        """
        Update the state history with the latest observation and action.

        Args:
            param (MPPIRunningParams): Current MPPI running parameters.
            current_state (jnp.ndarray): state at t (E, X)
            prev_action (jnp.ndarray): action at t-1 (E, A)

        Returns:
            MPPIRunningParams: Updated MPPI running parameters with new state history.
        """
        state_hist = param.state_hist
        state_hist = jnp.roll(state_hist, shift=-1, axis=0)
        state_hist = state_hist.at[-1, :, :self.params.num_obs].set(current_state)
        state_hist = state_hist.at[-2, :, self.params.num_obs:].set(prev_action)
        
        return MPPIRunningParams(
            a_mean_flattened = param.a_mean_flattened,
            a_cov_flattened = param.a_cov_flattened,
            prev_a_flattened = param.prev_a_flattened,
            state_hist = state_hist,
            key = param.key,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def debug_rollout(self, key, obs, a_list, running_params: MPPIRunningParams, static_features):
        """
        Perform a debug rollout to visualize the optimized trajectory.

        This function executes a single rollout using the current mean action sequence
        and returns the resulting state trajectory. It's useful for debugging and
        visualizing the behavior of the controller.

        Args:
            key (jax.random.PRNGKey): Random key for any stochastic operations.
            obs (jnp.ndarray): Current observation of the system state.
            a_list (jnp.ndarray): Mean action sequence to use for the rollout.
            running_params (MPPIRunningParams): Current MPPI running parameters.
            dynamic_params_tuple (Tuple): Parameters for the dynamics model.

        Returns:
            jnp.ndarray: Optimized state trajectory resulting from the rollout.
        """
        state_init = jnp.array(obs)
        action_expand_flattened = jnp.tile(jnp.expand_dims(a_list, 0), (self.params.n_rollouts, 1, 1))
        action_expand = action_expand_flattened.reshape(
            *action_expand_flattened.shape[:2], 
            self.params.num_entities, 
            self.params.num_actions
        )
        key, key2 = jax.random.split(key, 2)
        optim_traj = jnp.stack(self._get_rollout(key2, state_init, running_params.state_hist, action_expand, static_features, self.params.fix_history))[:, 0]
        return optim_traj
        
        
        
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs,
        goal_list, 
        running_params: MPPIRunningParams,
        static_features,
    ):
        """
        Execute the Model Predictive Path Integral (MPPI) control algorithm.

        This function performs the core MPPI computation, including:
        1. Sampling action trajectories
        2. Simulating rollouts using the dynamics model
        3. Evaluating costs for each trajectory
        4. Updating the control distribution based on trajectory costs

        Args:
            obs (jnp.ndarray): Current observation of the system state.
            goal_list (List[jnp.ndarray]): List of goal states for the trajectory.
            running_params (MPPIRunningParams): Current MPPI running parameters.
            static_features (jnp.ndarray): Parameters for the dynamics model.
            vis_optim_traj (bool, optional): Flag to visualize the optimized trajectory. Defaults to False.
            vis_all_traj (bool, optional): Flag to visualize all sampled trajectories. Defaults to False.

        Returns:
            Tuple: Contains the following elements:
                - jnp.ndarray: Optimal action to take at the current timestep.
                - MPPIRunningParams: Updated MPPI running parameters.
                - Dict: Additional information for debugging and visualization.
        """
        
        ## Note: 1. Sampling action trajectories
        key_use, self_key = jax.random.split(running_params.key, 2)
        key_use = jax.random.split(key_use, self.params.n_rollouts)
        
        def single_sample(key, traj_mean, traj_cov):
            keys = jax.random.split(key, self.params.h_knot)
            return jax.vmap(
                lambda key, mean, cov: jax.random.multivariate_normal(key, mean, cov)
            )(keys, traj_mean, traj_cov)

        a_mean_waypoint = running_params.a_mean_flattened[::self.params.num_intermediate]
        
        ## Spline interpolation
        for d in range(running_params.a_mean_flattened.shape[1]):
            a_mean_waypoint = a_mean_waypoint.at[:, d].set(
                self.u2node(running_params.a_mean_flattened[:, d])
            )
        
        
        a_cov_waypoint = running_params.a_cov_flattened[::self.params.num_intermediate]
        
        a_sampled_waypoint = jax.vmap(single_sample, in_axes=(0, None, None))( # (N, h_knot, action_dim)
            key_use, a_mean_waypoint, a_cov_waypoint,
        )
    
        ### Spline interpolation
        a_sampled_flattened = self.action_sampled_flattened.copy()
        for d in range(a_sampled_waypoint.shape[2]):
            a_sampled_flattened = a_sampled_flattened.at[:, :, d].set(
                self.node2u_vmap(a_sampled_waypoint[:, :, d])
            )
        

        a_sampled_raw = self.normalize_action(a_sampled_flattened)
        a_sampled_flattened = self.flattened_action_init_buf.copy()
        a_sampled_flattened = a_sampled_flattened.at[:, :self.params.delay, :].set(running_params.prev_a_flattened)
        a_sampled_flattened = a_sampled_flattened.at[:, self.params.delay:, :].set(a_sampled_raw)
        
        current_state = obs
        
        ## Note: 2. Simulating rollouts using the dynamics model
        self_key, key2 = jax.random.split(self_key, 2)
        a_sampled = a_sampled_flattened.reshape(*a_sampled_flattened.shape[:2], self.params.num_entities, self.params.num_actions)
        state_list = self._get_rollout(key2, current_state, running_params.state_hist, a_sampled, static_features)

        reward_rollout = self.get_reward(state_list, a_sampled, running_params.state_hist[-2, :, self.params.num_obs:], goal_list)
        cost_rollout = -reward_rollout
        cost_exp = jnp.exp(-(cost_rollout - jnp.min(cost_rollout)) / self.params.lam)
        weight = cost_exp / cost_exp.sum()


        a_sampled_flattened = a_sampled_flattened[:, self.params.delay:, :]
        
        ## Note: 3. Evaluating costs for each trajectory
        a_mean_flattened = jnp.sum(
            weight[:, None, None] * a_sampled_flattened, axis=0
        ) * self.params.gamma_mean + running_params.a_mean_flattened * (
            1 - self.params.gamma_mean
        )

        a_cov = jnp.sum(
                        weight[:, None, None, None] * ((a_sampled_flattened - a_mean_flattened)[..., None] * (a_sampled_flattened - a_mean_flattened)[:, :, None, :]),
                        axis=0,
                    ) * self.params.gamma_sigma + running_params.a_cov_flattened * (1 - self.params.gamma_sigma)
        
        u_flattened = a_mean_flattened[0]
        u = u_flattened.reshape(self.params.num_entities, self.params.num_actions)

        optim_traj = None
        action_expand_flattened = jnp.expand_dims(a_mean_flattened, 0)
        action_expand = action_expand_flattened.reshape(
            *action_expand_flattened.shape[:2], 
            self.params.num_entities, 
            self.params.num_actions
        )
        
        self_key, key2 = jax.random.split(self_key, 2)
        # (T_future, E, X)
        optim_traj = jnp.stack(self._get_rollout(key2, current_state, running_params.state_hist, action_expand, static_features))[0]
        
        prev_a_flattened = jnp.concatenate([running_params.prev_a_flattened[1:], a_mean_flattened[:1]], axis=0)         

        new_running_params = MPPIRunningParams(
            a_mean_flattened = jnp.concatenate([a_mean_flattened[1:], a_mean_flattened[-1:]], axis=0),
            a_cov_flattened = jnp.concatenate([a_cov[1:], a_cov[-1:]], axis=0),
            state_hist = running_params.state_hist,
            prev_a_flattened = prev_a_flattened,
            key = self_key,
        )

        info_dict = {
            'trajectory': optim_traj, 
            'action': u, 
            'a_mean_jnp': a_mean_flattened,
            'action_candidate': None, 'x_all': None, 'y_all': None,
            
            ### Note: Need to comment out the @jax.jit decorator for the following two lines to visualize the history
            #  'history': running_params.state_hist,
            #  'all_traj': state_list[:, best_100_idx],
        }
        
        return u,  new_running_params,  info_dict
