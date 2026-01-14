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
        return {
            "current_state": self.current_state,
            "static_features": self.static_features
        }

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
        
        self.current_state = self._get_current_state()

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

    def change_parameters(self, car_params: dict|None = None):
        if car_params:
            self.world.change_parameters(car_params)
            return car_params
        params = {}
        
        params["mass"] = self.generate_new_mass()
        params["com"] = self.generate_new_com()        
        params["friction"] = self.generate_new_friction() 
        params["wheel_parameters"] = self.generate_new_wheel_parameters()
        params["wheel_base"] = self.generate_new_wheel_base()
        params["wheel_track"] = self.generate_new_wheel_track()
        params["max_throttle"] = self.generate_new_max_throttle()        
        params["delay"] = self.generate_new_delay()        
        params["max_steer"] = self.generate_new_max_steering()        
        params["steer_bias"] = self.generate_new_steering_bias()
        self.world.change_parameters(params)
        return params

    # generate a new mass
    def generate_new_mass(self):   
        # print("[Warn] Car Mass Generation Not Defined for Simulator Type")
        # default_mass = 3.794137 # base mujoco mass
        default_mass = self.world.model.body_mass[self.world.root_id].copy()
        
        lower = default_mass * 0.7
        upper = default_mass * 1.3
        new_mass = np.random.uniform(lower, upper)

        return new_mass

    def generate_new_com(self):
        # print("[Warn] COM Generation Not Defined for Simulator Type")
        # default_com = np.array([-0.02323112, -0.00007926,  0.09058852]) # base mujoco COM
        default_com = self.world.model.body_ipos[self.world.root_id].copy()

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
    
    def generate_new_wheel_parameters(self):
        default_radius, default_width = self.world.model.geom_size[self.world.rim_geom_ids[0]]
        default_mass = self.world.model.body_mass[self.world.rim_body_ids[0]]

        radius = np.random.normal(default_radius, default_radius/4, 1)
        width = np.random.normal(default_width, default_width/4, 1)
        mass = np.random.normal(default_mass, default_mass/4, 1)
        return [radius, width, mass]
    
    def generate_new_wheel_base(self):
        """ Assume Four wheels with FL, FR, RL, RR order"""
        default_front = self.world.model.body_pos[self.world.knuckle_body_ids[0]][0]
        default_rear = self.world.model.body_pos[self.world.knuckle_body_ids[2]][0]

        front = np.random.normal(default_front, default_front/4, 1)
        rear = np.random.normal(default_rear, default_rear/4, 1)
        return [front, rear]
    
    def generate_new_wheel_track(self):
        """ Assume Four wheels with FL, FR, RL, RR order"""
        default_track = (
            self.world.model.body_pos[self.world.knuckle_body_ids[0]][1] - 
            self.world.model.body_pos[self.world.knuckle_body_ids[1]][1]
        )

        track = np.random.normal(default_track, default_track/4, 1)
        return track

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
    
    def generate_new_actuation_config(self):
        """
        Returns:
            mask: dict
                {
                'FL': [x, y, z, camber, throttle, steering],
                'FR': [x, y, z, camber, throttle, steering],
                'RL': [x, y, z, camber, throttle, steering],
                'RR': [x, y, z, camber, throttle, steering],
                }
        """
        wheels = ["FL", "FR", "RL", "RR"]

        # --- 1. Linear motions (shared across all wheels) ---
        x_move = np.random.choice([True, False])
        y_move = np.random.choice([True, False])
        z_move = np.random.choice([True, False])

        # --- 2. Drivetrain (throttle) ---
        drivetrain = np.random.choice(["4WD", "FWD", "RWD"])

        if drivetrain == "4WD":
            throttle = {"FL": True, "FR": True, "RL": True, "RR": True}
        elif drivetrain == "FWD":
            throttle = {"FL": True, "FR": True, "RL": False, "RR": False}
        else:  # RWD
            throttle = {"FL": False, "FR": False, "RL": True, "RR": True}

        # --- 3. Steering ---
        steering_mode = np.random.choice(["4WS", "FWS", "RWS"])

        if steering_mode == "4WS":
            steering = {"FL": True, "FR": True, "RL": True, "RR": True}
        elif steering_mode == "FWS":
            steering = {"FL": True, "FR": True, "RL": False, "RR": False}
        else:  # RWS
            steering = {"FL": False, "FR": False, "RL": True, "RR": True}

        # --- 4. Camber ---
        front_camber = np.random.choice([True, False])
        rear_camber = np.random.choice([True, False])

        camber = {
            "FL": front_camber,
            "FR": front_camber,
            "RL": rear_camber,
            "RR": rear_camber,
        }

        # --- 5. Assemble mask ---
        mask = {}
        for w in wheels:
            mask[w] = [
                x_move,          # x linear
                y_move,          # y linear
                z_move,          # z linear
                camber[w],       # camber
                throttle[w],     # throttle
                steering[w],     # steering
            ]

        return mask
