from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.interpolate import make_interp_spline
import math
import matplotlib.pyplot as plt

class AltPurePursuitController:
    DEFAULT = {
        'lowervel': -1,
        'uppervel': 1,
        'max_steering': None,
        'wheelbase': None,
        'totaltime': 1000,
    }
    def __init__(self, config={}):
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

        self._target_velocities = []
    
        self.prev_index = None
        self.prev_target_pos = [0, 0]

        self.first_index = None

        self.lookahead_distance = 4
        self.target_pos = []
        self.n_actions = 5
        self.last_n_actions = []

        self.generate_target_velocities()
        self.name = "pure_pursuit"

        self.threshold_violated = False
        self.alpha_threshold_max = np.deg2rad(50.0)
        self.alpha_threshold_min = np.deg2rad(30.0)
        self.guard_steer = 0.9


    # Pure Pursuit
    def get_target_pos(self, world, path, velocity = 1.0):
        xpos = world.pose[:2]

        pathlength = path.shape[0]
        search_window = pathlength//4
        xpos_x = xpos[0]
        xpos_y = xpos[1]

        direction = 1 if velocity >= 0 else -1

        # remember to reset the indexes 
        if self.prev_index is None:
            # get initial target point
            nearest_ind = self.calc_nearest_ind(xpos_x, xpos_y, path[:-1])
            self.prev_index = nearest_ind
            self.prev_target_pos = path[nearest_ind]
            return path[nearest_ind]

        for i in range(math.floor(self.prev_index), min(math.ceil(self.prev_index) + search_window, pathlength-1)):
            currPoint = path[i]
            nextPoint = path[i + 1]
            # if i == pathlength - 1:
            #     self.prev_index = 0
            d = nextPoint - currPoint #vector in direction of travel
            f = currPoint - xpos #vector to start segment
            r = self.lookahead_distance #lookahead distance
            a = np.dot(d, d)    
            b = 2 * np.dot(f, d)
            c = np.dot(f,f) - (r * r)
            discriminant = b * b - 4 * a * c
            if(discriminant > 0):
                discriminant = math.sqrt(discriminant) 
                t1 = (-b - discriminant) / (2 * a) 
                t2 = (-b + discriminant) / (2 * a)
                for t in (t1, t2):
                    idx = i + t
                    if not (0 <= t <= 1 and idx > self.prev_index):
                        continue

                    target_pos = currPoint + t * d

                    self.prev_target_pos = target_pos
                    self.prev_index = idx
                    return target_pos
                
        if math.ceil(self.prev_index) == pathlength-1:
            self.prev_index = 0 
            # self.lap_counter += 1
            # print("Target Velocity", self.target_vel)

        return self.prev_target_pos

    def calc_nearest_ind(self, xpos_x, xpos_y, path):
        """
        calc index of the nearest point to current position
        :param node: current information
        :return: index of nearest point
        """

        path_x = path[:,0]
        path_y = path[:,1]  
        dx = [xpos_x - x for x in path_x]
        dy = [xpos_y - y for y in path_y]

        ind = np.argmin(np.hypot(dx, dy))
        return ind

    def calc_distance(self, xpos_x, xpos_y, path_x, path_y):
        return math.hypot(xpos_x - path_x, xpos_y - path_y)

    def calculate_steering_angle(self, world, target_pos, velocity):
        curr_pos = world.pose
        yaw = world.rpy[2]
        dx = target_pos[0] - curr_pos[0]
        dy = target_pos[1] - curr_pos[1]
        lookahead_angle = np.arctan2(dy, dx)

        direction = np.sign(velocity)
        if direction == 0:
            direction = 1.0

        effective_yaw = yaw if direction > 0 else yaw + np.pi
        effective_yaw = np.arctan2(np.sin(effective_yaw), np.cos(effective_yaw))
        alpha = lookahead_angle - effective_yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        guard_steer = self.guard_steer * np.sign(alpha) * direction

        steer_rad = np.arctan2(2 * self.wheelbase * np.sin(alpha), self.lookahead_distance)
        pp_steer = np.clip(steer_rad/self.max_steering, -1., 1.)*direction

        if abs(alpha) > self.alpha_threshold_max:
            self.threshold_violated = True
            return guard_steer
        elif self.alpha_threshold_min <= abs(alpha) <= self.alpha_threshold_max:
            return guard_steer*0.5 if self.threshold_violated else pp_steer
        else:
            self.threshold_violated = False
            return pp_steer

    #TODO: Fix Naming: This actually generates target throttles
    def generate_target_velocities(self):
        samples = self.totaltime // 100
        mean = self.lowervel + (self.uppervel - self.lowervel) * 0.5
        std_dev = (self.uppervel - self.lowervel) / 4
        sampled_vels = np.random.normal(mean, abs(std_dev), samples)

        sampled_vels = np.clip(sampled_vels, self.lowervel, self.uppervel)
        x = np.linspace(0, self.totaltime, samples)
        spline = make_interp_spline(x, sampled_vels)
        x_total = np.arange(0, self.totaltime)
        target_velocities = spline(x_total)

        self._target_velocities = np.clip(target_velocities, self.lowervel, self.uppervel).copy()
        assert np.all(abs(self._target_velocities) >= abs(self.lowervel))
        
        # plt.plot(self._target_velocities)
        # plt.show()
  
    def get_control(self, t, world, path):
        throttle = self.target_velocities[t]
        self.target_pos = self.get_target_pos(world, path, throttle)
        assert np.all(abs(self.target_velocities) >= abs(self.lowervel))
        steer = self.calculate_steering_angle(world, self.target_pos, throttle)

        return [throttle, steer]
    
    def reset_controller(self):
        self.prev_index = None
        self.prev_target_pos = [0, 0]
        self.generate_target_velocities()
        
    @property
    def target_velocities(self):
        return deepcopy(self._target_velocities)