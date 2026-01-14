import glob
import pickle
from random import random
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle
from matplotlib.animation import FuncAnimation
from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
import os

# ---------- quaternion → yaw ----------
def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

# ---------- draw rotated rectangle ----------
def draw_rect(ax, center, w, h, yaw, color):
    cx, cy = center
    rect = Rectangle(
        (cx - w / 2, cy - h / 2),
        w, h,
        angle=np.degrees(yaw),
        rotation_point='center',
        fill=False,
        edgecolor=color,
        linewidth=2
    )
    ax.add_patch(rect)

# ---------- main visualization ----------
def visualize_timestep(state, action):
    fig, ax = plt.subplots(figsize=(6, 6))

    # ===== root =====
    root_pos = state[0, 0:2]
    root_quat = state[0, 3:7]
    yaw = quat_to_yaw(root_quat)

    # vehicle body
    body_length = 4.0
    body_width = 2.0
    draw_rect(ax, root_pos, body_length, body_width, yaw, "black")

    # ===== wheels =====
    wheel_w, wheel_h = 0.4, 0.8

    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    for i in range(1, 5):
        local_pos = state[i, 0:2]
        world_pos = root_pos + R @ local_pos
        draw_rect(ax, world_pos, wheel_w, wheel_h, yaw, "blue")

    # ===== action arrows =====
    # 평균 action (4개 바퀴)
    throttle = np.mean(action[1:, 4])
    steering = np.mean(action[1:, 5])

    # longitudinal (x-axis of vehicle)
    long_dir = R @ np.array([1.0, 0.0])
    lat_dir  = R @ np.array([0.0, 1.0])

    ax.arrow(
        root_pos[0], root_pos[1],
        long_dir[0] * throttle,
        long_dir[1] * throttle,
        width=0.05,
        color='red',
        label='Throttle'
    )

    ax.arrow(
        root_pos[0], root_pos[1],
        lat_dir[0] * steering,
        lat_dir[1] * steering,
        width=0.05,
        color='green',
        label='Steering'
    )

    # ===== plot config =====
    ax.set_aspect('equal')
    ax.set_xlim(root_pos[0] - 6, root_pos[0] + 6)
    ax.set_ylim(root_pos[1] - 6, root_pos[1] + 6)
    ax.set_title("Top-view Vehicle Visualization")
    ax.grid(True)

    plt.show()

def animate_dataset(states, actions, interval=100):
    fig, ax = plt.subplots(figsize=(6, 6))

    # ----- vehicle parameters -----
    body_length = 0.3
    body_width = 0.2
    wheel_w, wheel_h = 0.02, 0.04

    # ----- initialize patches -----
    body = Rectangle((0, 0), body_length, body_width,
                     fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(body)

    front_axle = Circle((0,0), 0.01, fill=True, color="black")
    ax.add_patch(front_axle)

    wheels = []
    throttle_arrows = []
    steering_arrows = []
    for _ in range(4):
        w = Rectangle((0, 0), wheel_h, wheel_w,
                      fill=False, edgecolor="blue")
        wheels.append(w)
        ax.add_patch(w)

        throttle_arrow = FancyArrow(0, 0, 0, 0, color="red", width=0.05)
        throttle_arrows.append(throttle_arrow)
        ax.add_patch(throttle_arrow)

        steering_arrow = FancyArrow(0, 0, 0, 0, color="green", width=0.05)
        steering_arrows.append(steering_arrow)
        ax.add_patch(steering_arrow)

    ax.set_aspect("equal")
    ax.grid(True)

    # ----- update function -----
    def update(frame):
        nonlocal throttle_arrows, steering_arrows

        state = states[frame]
        action = actions[frame]

        # === root ===
        root_pos = state[0, 0:2]
        root_quat = state[0, 3:7]
        yaw = quat_to_yaw(root_quat)

        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        # --- update body ---
        body.set_xy(root_pos - R @ np.array([body_length/2, body_width/2]))
        body.angle = np.degrees(yaw)

        # --- update wheels ---
        for i in range(4):
            local_pos = state[i+1, 0:2]
            local_quat = state[i+1, 3:7]
            wheel_pos = root_pos + R @ local_pos- R @ np.array([wheel_h/2, wheel_w/2])
            wheel_yaw = quat_to_yaw(local_quat)
            wheels[i].set_xy(wheel_pos)
            wheels[i].angle = np.degrees(yaw + wheel_yaw)

            # === action arrows ===
            throttle = action[i+1, 4]
            steering = action[i+1, 5]

            long_dir = R @ np.array([wheel_h, 0.0])
            lat_dir  = R @ np.array([0.0, wheel_h])

            throttle_arrows[i].set_data(
                x=wheel_pos[0], y=wheel_pos[1],
                dx=long_dir[0] * throttle,
                dy=long_dir[1] * throttle,
                width=0.005, head_width=0.01, head_length = wheel_h*throttle*0.1
            )
            steering_arrows[i].set_data(
                x=wheel_pos[0], y=wheel_pos[1],
                dx=lat_dir[0] * steering*2,
                dy=lat_dir[1] * steering*2,
                width=0.005, head_width=0.01, head_length = wheel_h*throttle*0.1
            )

        front_axle_pos = tuple([
            (wheels[0].get_xy()[0] + wheels[1].get_xy()[0]) / 2,
            (wheels[0].get_xy()[1] + wheels[1].get_xy()[1]) / 2
        ])
        front_axle.set_center(
            front_axle_pos
        )

        # --- camera follows vehicle ---
        ax.set_xlim(root_pos[0] - 0.6, root_pos[0] + 0.6)
        ax.set_ylim(root_pos[1] - 0.6, root_pos[1] + 0.6)

        ax.set_title(f"Timestep: {frame}")

        return body, *wheels, *throttle_arrows, *steering_arrows

    ani = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.show(block=True)
    return


if __name__ == "__main__":
    dataset_path = os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_debugging')
    dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl'))
    dataset_name = os.path.basename(dataset_files[random.randint(0, len(dataset_files)-1)])

    with open(os.path.join(dataset_path, dataset_name), "rb") as f:
        dataset = pickle.load(f)

    # 예: 첫 timestep
    states = dataset.data_logs["state"]
    actions = dataset.data_logs["action"]

    animate_dataset(states, actions, interval=100)