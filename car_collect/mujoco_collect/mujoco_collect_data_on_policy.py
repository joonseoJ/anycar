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
from car_planner.global_trajectory import GlobalTrajectory
# from car_planner.track_generation_realistic import change_track
from car_dynamics import MUJOCO_MODEL_DIR

from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_dynamics.models_jax import DynamicsJax
from car_foundation import CAR_FOUNDATION_MODEL_DIR

from car_ros2.utils import load_mppi_params, load_dynamic_params

import jax

import faulthandler
faulthandler.enable()

def empty_fn():
    ...

def log_data(dataset, obs):
    dataset.data_logs["history"].append(obs['history'])
    dataset.data_logs["static_features"].append(obs['static_features'])
    dataset.data_logs["current_state"].append(obs['current_state'])

def get_rollout_fn(use_ray):
    if use_ray:
        return ray.remote(rollout)
    else:
        return rollout

def rollout(id, simend, render, debug_plots, datadir, mppi: MPPIController, key_i):
    tic = time.time()

    dataset = CarDataset()
    env = MuJoCoCar({'is_render': render}) # create the simulation environment
    env.reset()

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
    print("Car Params", dataset.car_params)

    #choose to change parameters or not
    env.world.change_parameters(dataset.car_params)

    direction = np.random.choice([-1, 1])
    scale = int(np.random.uniform(1, 5))
    trajectory = change_track(scale, direction)
    global_planner = GlobalTrajectory(trajectory)
    env.world.trajectory = trajectory

    # MPPI Controller setting
    model_params = load_dynamic_params()
    mppi_running_params = mppi.get_init_params()
    key_i, key2 = jax.random.split(key_i)
    mppi_running_params = MPPIRunningParams(
        a_mean = mppi_running_params.a_mean,
        a_cov = mppi_running_params.a_cov,
        prev_a = mppi_running_params.prev_a,
        state_hist = mppi_running_params.state_hist,
        key = key2,
    )
    
    
    # tuned kp and kd
    kp = np.random.uniform(6, 10)
    kd = np.random.uniform(0.5, 1.5)
    
    last_err_vel = 0.
    is_terminate = False 
    clipped = 0
    actions = []
    history_length = env.observation_space['history'].shape[0]
    for t in tqdm(range(simend)):
        state = env.obs_state()
        
        if t == 0:
            for _ in range(history_length):
                mppi_running_params = mppi.feed_hist(mppi_running_params, state, np.array([0., 0.]))

        target_pos_arr, frenet_pose = global_planner.generate(state[:5], env.sim.params.DT, (mppi_params.h_knot - 1) * mppi_params.num_intermediate + 2 + mppi_params.delay, True)
        target_pos_list = np.array(target_pos_arr)
        target_pos_tensor = jnp.array(target_pos_arr)
        dynamic_params_tuple = (model_params.LF, model_params.LR, model_params.MASS, model_params.DT, model_params.K_RFY, model_params.K_FFY, model_params.Iz, model_params.Ta, model_params.Tb, model_params.Sa, model_params.Sb, model_params.mu, model_params.Cf, model_params.Cr, model_params.Bf, model_params.Br, model_params.hcom, model_params.fr)
        
        action, mppi_running_params, mppi_info = mppi(state,target_pos_tensor, mppi_running_params, dynamic_params_tuple, vis_optim_traj=True,)
        mppi_running_params = mppi.feed_hist(mppi_running_params, state, action)
        
        obs, reward, done, info = env.step(np.array(action))

        log_data(dataset, obs)

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if np.abs(env.world.rpy[0]) > 0.05 or np.abs(env.world.rpy[1]) > 0.05:
            is_terminate = True
            break
    
    if not is_terminate:
        # dataset.data_logs["lap_end"][-1] = 1 
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(id) + '_' + str(now) + ".pkl"
        filepath = os.path.join(datadir, file_name)
        
        for key, value in dataset.data_logs.items():
            dataset.data_logs[key] = np.array(value)

        with open(filepath, 'wb') as outp: 
            pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

        print("Saved Data to:", filepath)

        # if debug_plots:
        #     actions = np.array(actions)
        #     print(actions.shape)
        #     plt.plot(actions, label = "actual command")
        #     # plt.plot(actions[:, 1], label = "steering")
        #     # plt.plot(ppcontrol.target_velocities, label="targetvel")
        #     plt.legend()
        #     plt.show()
        #     plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='Trajectory')
        #     plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
        #     plt.show()
        
        dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)
    return not is_terminate

if __name__ == "__main__":
    jax_key = jax.random.PRNGKey(0)
    render = False
    debug_plots = False
    simend = 20000
    episodes = 100
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "mujoco_sim_debugging")
    resume_model_name = "2026-01-07T14:11:07.903-jax_model_checkpoint"
    resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, "checkpoint_0")
    os.makedirs(data_dir, exist_ok=True)

    mppi_params = load_mppi_params()
    model_params = load_dynamic_params()
    jax_key, key2 = jax.random.split(jax_key)
    dynamics = DynamicsJax({'model_path': resume_model_folder_path})
    mppi_rollout_fn = rollout_fn_jax(dynamics)
    mppi = MPPIController(
        mppi_params, mppi_rollout_fn, empty_fn, key2
    )

    num_success = 0
    start = time.time()
    if render or debug_plots:
        rollout_fn = get_rollout_fn(False)
        for i in range(episodes):
            jax_key, key2 = jax.random.split(jax_key)
            ret = rollout_fn(i, simend, render, debug_plots, data_dir, mppi, key2)
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
            jax_key, key2 = jax.random.split(jax_key)
            ret.append(rollout_fn.remote(i, simend, render, debug_plots, data_dir, mppi, key2))
            print(f"Episode {i} Appended")
        output = ray.get(ret)
        # print(output)
        for ifsuccess in output:
            if ifsuccess:
                num_success+=1
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")    
