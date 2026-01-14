from car_foundation import CAR_FOUNDATION_DATA_DIR
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import ray
import random
import time
import math
import datetime
from tqdm import tqdm
from car_dynamics.envs.mujoco_sim.cam_utils import *
from car_dataset import CarDataset
from car_planner.track_generation import change_track
# from car_planner.track_generation_realistic import change_track
from car_dynamics import MUJOCO_MODEL_DIR

from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
from car_dynamics.models_jax import DynamicsJax
from car_foundation import CAR_FOUNDATION_MODEL_DIR

import jax
import jax.numpy as jnp

import faulthandler
faulthandler.enable()

def visualize_diff(diff):
    plt.figure()
    plt.imshow(diff)
    plt.colorbar()
    plt.title("Difference between NumPy and JAX")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

    print("max diff:", diff.max())
    print("mean diff:", diff.mean())

def get_rollout_fn(use_ray):
    if use_ray:
        return ray.remote(rollout)
    else:
        return rollout

def rollout(id, simend, render, debug_plots, datadir):
    tic = time.time()

    dataset = CarDataset()
    env = MuJoCoCar({'is_render': render}) # create the simulation environment
    obs = env.reset()

    # set the simulator
    dataset.car_params["sim"] = env.name
    # set the wheelbase
    dataset.car_params["wheelbase"] = env.wheelbase
    # generate a new mass
    dataset.car_params["mass"] = env.generate_new_mass()
    # generate a new com
    dataset.car_params["com"] = env.generate_new_com()
    # generate a new friction
    dataset.car_params["friction"] = env.generate_new_friction()
    # generate new max throttle
    dataset.car_params["max_throttle"] = env.generate_new_max_throttle()
    # generate new delay
    dataset.car_params["delay"] = env.generate_new_delay()
    # generate new max steering
    dataset.car_params["max_steer"] = env.generate_new_max_steering()
    # generate new steering bias
    dataset.car_params["steer_bias"] = env.generate_new_steering_bias()

    # fine tuning data collection
    # dataset.car_params['mass'] = 3.794
    # dataset.car_params['friction'] = 1.
    # dataset.car_params['max_throttle'] = 10.
    # dataset.car_params['delay'] = 4

    print("Car Params", dataset.car_params)

    #choose to change parameters or not
    env.world.change_parameters(dataset.car_params)

    direction = np.random.choice([-1, 1])
    scale = int(np.random.uniform(1, 5))

    #wenli where did you get the lower vel from?
    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': simend,
        'lowervel': 0.7, #actual min vel
        'uppervel': 2., #actual max vel   
        'max_steering': env.world.max_steer    
    })

    controller = ppcontrol #all_controllers[np.random.choice([0, 1])]
    trajectory = change_track(scale, direction)
    env.world.trajectory = trajectory
    
    # tuned kp and kd
    kp = np.random.uniform(6, 10)
    kd = np.random.uniform(0.5, 1.5)
    
    last_err_vel = 0.
    is_terminate = False 
    clipped = 0
    actions = []


    resume_model_name = "2026-01-14T11:14:00.296-model_checkpoint"
    resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name)
    dynamics = DynamicsJax({
        'model_path': resume_model_folder_path,
        'model_dim': 128,
        'state_dim': 13,
        'action_dim': 6,
        'static_dim': 9,
        'history_dim': 19,
        'history_length': 100,
        'num_entities': 5,
        'num_heads': 4,
        'num_layers': 2
    })
    jax_key = jax.random.PRNGKey(0)
    for t in range(simend):
        if t == 0:
            state = obs['current_state']
            history = jnp.tile(state[None, None, :, :], (1, dynamics.params['history_length'], 1, 1))
            history = jnp.concatenate([history, jnp.zeros((*history.shape[:3], 6))], axis=-1)
                

        action = controller.get_control(t, env.world, trajectory)
        if controller.name == "pure_pursuit":

            target_vel = action[0]

            action[0] = kp * (target_vel - env.world.lin_vel[0]) + kd * ((target_vel - env.world.lin_vel[0]) - last_err_vel)

            action[0] /= env.world.max_throttle # normalize to -1, 1
            
            # print(action[0])
            last_err_vel = target_vel - env.world.lin_vel[0]
        else:
            raise ValueError(f"Unknown Controller: {controller.name}")

        action_matrix = np.zeros(env.action_space.shape)
        action_matrix[:,4] = action[0]
        action_matrix[:,5] = action[1]
        #TODO Fix this, the action is getting clipped like 25% of the time
        action_matrix = np.clip(action_matrix, env.action_space.low, env.action_space.high)
        # actions.append(action[0]) 
        obs, reward, done, info = env.step(action_matrix)
        state = obs['current_state']


        jax_key, key2 = jax.random.split(jax_key)
        state_nn = dynamics.model_state.apply_fn(
            {'params': dynamics.model_state.params},
            history,
            obs['static_features'][None],
            action_matrix[None,None],
            rngs=key2,
        )
        state_nn = np.array(state_nn[0, 0,:])
        diff = np.abs(state - state_nn)

        visualize_diff(diff)

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if np.abs(env.world.rpy[0]) > 0.05 or np.abs(env.world.rpy[1]) > 0.05:
            is_terminate = True
            break
   
        current_history = jnp.concatenate(
            [state, action_matrix], axis=-1
        )
        history = jnp.concatenate(
            [history[:, 1:], current_history[None, None, :, :]],
            axis=1
        )
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)
    return not is_terminate

if __name__ == "__main__":

    render = True
    debug_plots = False
    simend = 20000
    episodes = 100
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "mujoco_sim_debugging")
    os.makedirs(data_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    if render or debug_plots:
        rollout_fn = get_rollout_fn(False)
        for i in range(episodes):
            ret = rollout_fn(i, simend, render, debug_plots, data_dir)
            print(f"Episode {i} Complete")
            if ret:
                num_success += 1
        
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")   
    else:
        assert(render == False and debug_plots == False)
        rollout_fn = get_rollout_fn(True)
        # Let's start Ray
        ray.init()
        ret = []
        for i in range(episodes):
            ret.append(rollout_fn.remote(i, simend, render, debug_plots, data_dir))
            print(f"Episode {i} Appended")
        output = ray.get(ret)
        # print(output)
        for ifsuccess in output:
            if ifsuccess:
                num_success+=1
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")    
