from copy import deepcopy
from typing import Union
import gym
from gym import spaces
import numpy as np

from termcolor import colored
from car_dynamics.envs.mujoco_sim import World
from scipy.spatial.transform import Rotation as R
from collections import deque

class MuJoCoCar(gym.Env):

    DEFAULT = {
        'max_step': 100,
        'dt': 0.02,
        'is_render': False,
        'delay': 0,

        'state_dim': 13,
        'static_dim': 9,
        'action_dim_per_entity': 6,
        'num_entities': 5,
        'history_dim': 19,
        'history_length': 100,

        'wheel_configs': None
    }


    def __init__(self, config: dict):
        super(MuJoCoCar, self).__init__()
        
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

        if not self.wheel_configs:
            self.wheel_configs = [
                {"pos":"0.1385  0.115  0.0488",  "mask":[False, False, False, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
                {"pos":"0.1385 -0.115  0.0488",  "mask":[False, False, False, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
                {"pos":"-0.158  0.115  0.0488",  "mask":[False, False, False, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
                {"pos":"-0.158 -0.115  0.0488",  "mask":[False, False, False, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
            ]
        self.num_wheels = len(self.wheel_configs)
        
        self.world = World({'is_render': self.is_render, 'wheel_configs': self.wheel_configs})

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.num_entities, self.action_dim_per_entity), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            # (N, self.num_entities, self.history_dim): History of dynamic states for [Root, W1, W2, W3, W4]
            # State and Applied action at t = T-1, ..., T-N+1
            "history": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.history_length, self.num_entities, self.history_dim), 
                dtype=np.float32
            ),
            # State of entites at t=T
            "current_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_entities, self.state_dim)
            ),
            # (self.num_entities, self.static_dim): Static features per entity
            "static_features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.num_entities, self.static_dim), 
                dtype=np.float32
            )
        })

        self.history_buffer = deque(maxlen=self.history_length)
        self.static_features = self._build_static_features()
        self.current_state = np.zeros((self.num_entities, self.state_dim), dtype=np.float32)

        self._step = None
        
        self.action_buffer = []
        
        self.target_velocity = 0

        self.wheelbase = 0.2965

        self.name = "mujoco"

        self.reset()

    def _build_static_features(self):
        """
        build static features for each entity.
        """
        features = np.zeros((self.num_entities, self.static_dim), dtype=np.float32)
        
        for i, config in enumerate(self.wheel_configs):
            idx = i + 1
            features[idx, :6] = config["mask"]
            features[idx, 6] = config["radius"]
            features[idx, 7] = config["width"]
            features[idx, 8] = config["mass"]

        return features

    def obs_state(self):        
        history = np.array(self.history_buffer, dtype=np.float32)
        
        return {
            "history": history,
            "current_state": self.current_state,
            "static_features": self.static_features
        }
    
    def _record_history(self, action=None):
        """
        Append (self.num_entities, self.history_dim) shaped state matrix for all entities to history buffer
        Row 0: Root State (World Frame)
        Row 1~4: Wheel States (Root Relative Frame)
        Each Col: 7 (Pose) + 6 (Twist) + 6 (Action) at t
        
        **Caution 1** This function should be called before action is applied
        **Caution 2** history[t][pose, twist] are not a result of history[t][action]
        """
        if action is None:
            action = np.zeros((self.num_entities, self.action_dim_per_entity), dtype=np.float32)
        
        state_matrix = self._get_current_state()
        history_matrix = np.concatenate((state_matrix, action), axis=1)

        self.history_buffer.append(history_matrix)

    def _get_current_state(self):
        state_matrix = np.zeros((self.num_entities, self.state_dim), dtype=np.float32)

        root_pos = self.world.pose
        root_quat = self.world.orientation
        root_lin_vel = self.world.lin_vel
        root_ang_vel = self.world.ang_vel
        
        state_matrix[0, 0:3] = root_pos
        state_matrix[0, 3:7] = root_quat
        state_matrix[0, 7:10] = root_lin_vel
        state_matrix[0, 10:13] = root_ang_vel

        for i in range(self.num_wheels):
            wheel_pos = self.world.wheel_pos(i)
            wheel_quat = self.world.wheel_quat(i)
            wheel_lin_vel = self.world.wheel_lin_vel(i)
            wheel_ang_vel = self.world.wheel_ang_vel(i)

            state_matrix[i+1, 0:3] = wheel_pos
            state_matrix[i+1, 3:7] = wheel_quat
            state_matrix[i+1, 7:10] = wheel_lin_vel
            state_matrix[i+1, 10:13] = wheel_ang_vel
        
        return state_matrix

    def reset(self):
        self._step = 0.
        self.target_velocity = 0.
        self.world.reset()
        # for _ in range(20):
        #     self.step(np.array([-1, 0.]))
        self.action_buffer = []
        for _ in range(self.delay):
            self.action_buffer.append(np.array([0., 0.], dtype=np.float32))
        
        self.history_buffer.clear()
        self.current_state = self._get_current_state()

        for _ in range(self.history_length):
            self._record_history()

        return self.obs_state()

    def reward(self,):
        return .0
    
    def step(self, action_):
        action_ = np.array(action_, dtype=np.float32)
        self.action_buffer.append(action_)
        self._step += 1
        action = self.action_buffer[0].copy()
        assert action.dtype == np.float32
        action = np.clip(action, self.action_space.low, self.action_space.high) #clip to -1, 1

        num_steps = int(self.dt / self.world.dt)

        # scale acceleration command 
        throttle = action[:, 4] * self.world.max_throttle #scale it to real values
        steer = action[:, 5] * self.world.max_steer  + self.world.steer_bias # scale it to real values
        for _ in range(num_steps):
            #calculate the target vel from throttle and previous velocity.
            self.target_velocity = throttle * self.world.dt + self.target_velocity
            action[:, 4] = self.target_velocity
            action[:, 5] = steer
            action[3:5, 5] *= -1

            self._record_history(action)
            self.world.step(action)
            self.current_state = self._get_current_state()

        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False
            
        if self.is_render:
            self.render()

        self.action_buffer.pop(0)
        return self.obs_state(), reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            self.world.render()

    def change_parameters(self, car_params: dict):
        self.world.change_parameters(car_params)

    # generate a new mass
    def generate_new_mass(self):   
        # print("[Warn] Car Mass Generation Not Defined for Simulator Type")
        default_mass = 3.794137 # base mujoco mass
        
        lower = default_mass * 0.7
        upper = default_mass * 1.3
        new_mass = np.random.uniform(lower, upper)

        return new_mass

    def generate_new_com(self):
        # print("[Warn] COM Generation Not Defined for Simulator Type")
        default_com = np.array([-0.02323112, -0.00007926,  0.09058852]) # base mujoco COM
        lower = default_com - 0.05 #5 cm range
        upper = default_com + 0.05

        new_com = np.random.uniform(lower, upper)

        return new_com
    
    def generate_new_friction(self):
        # print("[Warn] Friction Generation Not Defined for Simulator Type")

        default_friction = 1. #base mujoco friction
        lower = default_friction * 0.5
        upper = default_friction * 1.1
        friction = np.random.uniform(lower, upper)
        new_friction = np.array([friction, 0.005, 0.0001]) #static, torsional, rolling         

        return new_friction

    def generate_new_delay(self):
        lower = 0
        upper = 6
        new_delay = int(np.random.uniform(lower, upper))
        return new_delay
   
    def generate_new_max_throttle(self):
        # print("[Warn] Max Throttle Generation Not Defined for Simulator Type")
        lower = 2
        upper = 8
        max_thr = np.random.uniform(lower, upper)
        return max_thr
    
    def generate_new_max_steering(self):
        # print("[Warn] Max Steering Generation Not Defined for Simulator Type")
        lower = 0.15
        upper = 0.36
        max_steer = np.random.uniform(lower, upper)
        return max_steer
        # clip steering at random value
        # add bias to steering
        # 
        
    def generate_new_steering_bias(self):
        lower = 0.0
        upper = 0.01
        bias = np.random.uniform(lower, upper)
        return bias
        
    def generate_new_slope(self):
        # TODO not implemented right now
        print("[Warn] Slope Generation Not Defined for Simulator Type")
        pass
    
    
    