import os
import tempfile
import pickle
import numpy as np
import mujoco as mj
from copy import deepcopy
from mujoco.glfw import glfw
from .cam_utils import update_camera
from scipy.spatial.transform import Rotation as R


def initialize_environment(xml_path, timestep, render):
    
    # For callback functions
    global button_left 
    global button_middle
    global button_right
    global lastx
    global lasty

    #get the full path
    # dirname = os.path.dirname(__file__)
    # abspath = os.path.join(dirname + "/" + xml_path)
    # xml_path = abspath

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)                # MuJoCo data   
    cam = mj.MjvCamera()                        # Abstract camera
    opt = mj.MjvOption() 
    model.opt.timestep = timestep

    if render:  # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        window = glfw.create_window(1000, 600, "Demo", None, None) # visualization options  
        glfw.make_context_current(window)
        glfw.swap_interval(1)   
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)     
    else:
        scene = None
        context = None
        window = None
                         

    # print("Model Update Rate in Hz: ", 1/model.opt.timestep)
    # print("Model Geoms: ", [model.geom(i).name for i in range(model.ngeom)])
    # print("Body Geoms: ", [model.body(i).name for i in range(model.nbody)])

    return model, data, cam, opt, scene, context, window

def generate_xml(wheel_configs) -> str:
        """
        Generates a valid MuJoCo MJCF XML string for a car
        Wheel_configs: "pos": f"{x} {y} {z}", "mask": 6 boolean list
        Returns the path to the temporary file.
        """
        # Basic MJCF Header
        mjcf = f"""
        <mujoco model="custom_car">
            <compiler angle="radian"/>
            <option integrator="RK4" timestep="0.01"/>
            <asset>
                <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.26 0.12 0.36" rgb2="0.23 0.09 0.33"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
                <mesh name="chassis" scale=".01 .006 .0015"
                vertex=" 9   2   0
                        -10  10  10
                        9  -2   0
                        10  3  -10
                        10 -3  -10
                        -8   10 -10
                        -10 -10  10
                        -8  -10 -10
                        -5   0   20"/>
            </asset>

            <!-- TODO: set physical parametrers (frictionloss, damping, armature, range) properly-->
            <default>
                <default class="chassis_part">
                    <geom contype="0" conaffinity="0" group="1" rgba="1 1 1 1"/>
                </default>
                <default class="p_x">
                    <joint type="slide" axis="1 0 0" limited="true" range="-0.1 0.1" 
                        stiffness="1500" damping="100" frictionloss="0.1" armature="0.01"/>
                </default>
                <default class="p_y">
                    <joint type="slide" axis="0 1 0" limited="true" range="-0.1 0.1" 
                        stiffness="1500" damping="100" frictionloss="0.1" armature="0.01"/>
                </default>
                <default class="suspension">
                    <joint type="slide" axis="0 0 1" limited="true" range="-0.05 0.05" 
                        stiffness="1500" damping="100" frictionloss="0.1" armature="0.01"/>
                </default>
                <default class="camber">
                    <joint type="hinge" axis="1 0 0" limited="true" frictionloss="0.01" damping="0.001" armature="0.0002" range="-0.1 0.1"/>
                </default>
                <default class="throttle">
                    <joint type="hinge" axis="0 0 1" limited="false" frictionloss="0.001" damping="0.01" armature="0.01"/>
                </default>
                <default class="steering">
                    <joint type="hinge" axis="0 0 1" limited="true" frictionloss="0.01" damping="0.001" armature="0.0002" range="-0.38 0.38"/>
                </default>
            </default>

            <worldbody>
                <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
                <geom type="plane" size="300 300 .01" material="grid"/>

                <body name="root" pos="0.0 0.0 0.0" euler="0.0 0.0 0.0">
                    <camera name="track" mode="trackcom" pos="0 -2 2" xyaxes="1 0 0 0 0.7 0.7"/>
                    <joint type="free"/>
                    <site name="root_site" pos="0.0 0.0 0.0"/>
                    <geom name="root" type="mesh" mesh="chassis" pos="0 0 0.094655"/>
                    <inertial pos="0 0 0" mass="3.542137" diaginertia="0.02 0.02 0.02"/>
        """
        
        # Dynamically add wheels
        for i, wheel_config in enumerate(wheel_configs):
            mjcf += f"""
                    <body name="wheel_{i}_knuckle" pos="{wheel_config["pos"]}">
                        <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
                        <site name="wheel_{i}_knuckle_site" pos="0 0 0" size="0.01" rgba="0 0 0 0.1"/>
                    """
            if wheel_config["mask"][0]:
                mjcf += f"""
                        <joint class="p_x" name="p_x_{i}"/>
                        """
            if wheel_config["mask"][1]:
                mjcf += f"""
                        <joint class="p_y" name="p_y_{i}"/>
                        """
            # if wheel_config["mask"][2]:
            mjcf += f"""
                    <joint class="suspension" name="suspension_{i}"/>
                    """
            if wheel_config["mask"][3]:
                mjcf += f"""
                        <joint class="camber" name="camber_{i}"/>
                        """
            if wheel_config["mask"][5]:
                mjcf += f"""
                        <joint class="steering" name="steering_{i}"/>
                        """
            mjcf += f"""
                        <body name="wheel_{i}_rim" pos="0 0 0" zaxis="0 1 0">
                            <joint class="throttle" name="throttle_{i}"/>
                            <geom name="wheel_{i}_rim" type="cylinder" size="{wheel_config["radius"]} {wheel_config["width"]}" mass="{wheel_config["mass"]}" rgba=".5 .5 1 1" friction="1.2 0.005 0.0001" contype="1" conaffinity="1"/>
                            <site name="wheel_{i}_rim_site" pos="0 0 0" type="box" size=".006 .03 .015" rgba="1 0 0 1"/>
                        </body>
                    </body>
                    """

        mjcf += """
                </body>
            </worldbody>
            <actuator>
                <!-- TODO: set physical parametrers (k_p, k_v, gear, ...) properly-->
        """
        # Add actuators (motors) for each wheel
        for i, wheel_config in enumerate(wheel_configs):
            if wheel_config["mask"][0]:
                mjcf += f"""
                        <motor name="p_x_{i}" joint="p_x_{i}" gear="1" ctrllimited="true" ctrlrange="-200 200"/>
                        """
            if wheel_config["mask"][1]:
                mjcf += f"""
                        <motor name="p_y_{i}" joint="p_y_{i}" gear="1" ctrllimited="true" ctrlrange="-200 200"/>
                        """
            if wheel_config["mask"][2]:
                mjcf += f"""
                        <motor name="sus_{i}" joint="suspension_{i}" gear="1" ctrllimited="true" ctrlrange="-200 200"/>
                        """
            if wheel_config["mask"][3]:
                mjcf += f"""
                        <position class="camber" kp="25.0" name="camber_{i}" joint="camber_{i}" ctrlrange="-1 1" ctrllimited="true"/>
                        """
            if wheel_config["mask"][4]:
                mjcf += f"""
                        <velocity name="throttle_{i}" joint="throttle_{i}" kv="100" gear="0.04" forcelimited="true" forcerange="-500 500"/>  
                        """
            if wheel_config["mask"][5]:
                mjcf += f"""
                        <position class="steering" kp="25.0" name="steering_{i}" joint="steering_{i}" ctrlrange="-1 1" ctrllimited="true"/>
                        """
            
        mjcf += """
            </actuator>
            <sensor>
                <velocimeter name="root_lin_vel" site="root_site" />
                <gyro name="root_ang_vel" site="root_site" />
                <accelerometer name="root_acc" site="root_site" />
        """
        # Add sensors for each wheel
        for i, wheel_config in enumerate(wheel_configs):
            mjcf += f"""
                <framepos      name="wheel_{i}_pos"     objtype="site" objname="wheel_{i}_knuckle_site" reftype="body" refname="root"/> 
                <framequat     name="wheel_{i}_quat"    objtype="site" objname="wheel_{i}_knuckle_site" reftype="body" refname="root"/> 
                <framelinvel   name="wheel_{i}_lin_vel" objtype="site" objname="wheel_{i}_knuckle_site" reftype="body" refname="root"/> 
                <frameangvel   name="wheel_{i}_ang_vel" objtype="site" objname="wheel_{i}_rim_site"     reftype="body" refname="wheel_{i}_knuckle"/> 
            """
        
        mjcf += """
            </sensor>
        </mujoco>
        """
        
        # Save to temp file
        fd, path = tempfile.mkstemp(suffix=".xml", text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(mjcf)
        return path


class World:
    
    DEFAULT = {
        'timestep': 0.001,
        'is_render': False,
        # 'xml_path': 'models/one_turbo_slope.xml',
        'xml_path': None,
        'max_throttle': 8.0,
        'max_steer': 0.36,
        'steer_bias': 0,
        'wheel_configs': None
    }
    
    def __init__(self, config={}):
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
        
        
        self.wheel_masks = np.array([wheel_config['mask'] for wheel_config in self.wheel_configs])
        if not self.xml_path:
            self.xml_path = generate_xml(self.wheel_configs)
        model, data, cam, opt, scene, context, window  = initialize_environment(self.xml_path, self.timestep, self.is_render)
        
        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scene = scene
        self.context = context
        self.window = window
        
        self.dt = self.model.opt.timestep
        self.warmup_steps = int(2. / self.timestep)

        self.root_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "root")
        self.root_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "root")
        self.num_wheels = len(self.wheel_configs)
        
        self.knuckle_body_ids = []
        self.rim_body_ids = []
        self.rim_geom_ids = []
        for i in range(self.num_wheels):
            self.knuckle_body_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"wheel_{i}_knuckle"))
            self.rim_body_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"wheel_{i}_rim"))
            self.rim_geom_ids.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"wheel_{i}_rim"))
        
        
    def reset(self, ):
        # self.model.body_pos[1] = np.hstack((trajectory[0],[0]))
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        for _ in range(self.warmup_steps):
            self.data.ctrl[:] = 0.0
            mj.mj_step(self.model, self.data)
            # print("height", self.pose[2])
        
    def step(self, actions):
        """Step the simulation forward by one timestep
            map controller output [-1, 1]
            - to [0, max_speed] for throttle
            - to [-max_steering, max_steering] for steering

        Args:
            actions (_type_): _description_
        """
        # # self.data.ctrl[1] = actions[0] * (self.max_speed / 2) + (self.max_speed / 2)
        # self.data.ctrl[1] = actions[0] # mapping throttle to correct control
        # # self.data.ctrl[0] = actions[1] * self.max_steering
        # self.data.ctrl[0] = actions[1] # mapping steering to correct control

        filtered_action = actions[1:,:]
        filtered_action = filtered_action[self.wheel_masks]
        self.data.ctrl[:] = filtered_action

        mj.mj_step(self.model, self.data)
        
    def render(self, mode='human'):
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        update_camera(self.cam, self.data.geom_xpos[1])
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        self._render_trajectory()
        
        mj.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def _render_trajectory(self):
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            return
        wps = self.trajectory
        N = wps.shape[0]
        
        # 선의 두께 (미터 단위) 및 높이 설정
        line_width = 0.05 
        z_height = 0.05 # 바닥(0.0)보다 살짝 띄워야 보임

        for i in range(N - 1):
            # Scene의 최대 지오메트리 개수를 넘으면 그리기 중단 (에러 방지)
            if self.scene.ngeom >= self.scene.maxgeom:
                break
            
            # 현재 그릴 점 두 개 (시작점, 끝점) + Z축 추가
            p1 = np.array([wps[i,0 ],   wps[i, 1],   z_height])
            p2 = np.array([wps[i+1, 0], wps[i+1, 1], z_height])

            # 빈 지오메트리 슬롯 가져오기
            geom = self.scene.geoms[self.scene.ngeom]
            self.scene.ngeom += 1 # 사용한 슬롯 개수 증가

            # 지오메트리 초기화 (색상: RGBA)
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_CAPSULE, # 선보다 캡슐이 훨씬 잘 보입니다
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 1.0]) # 빨간색
            )

            # 두 점을 잇는 커넥터 설정
            mj.mjv_connector(
                geom,
                mj.mjtGeom.mjGEOM_CAPSULE,
                line_width,
                p1,
                p2
            )

    @property
    def pose(self):
        """global position [x, y, z] of the car

        Returns:
            np.array: [x, y, z]
        """
        return self.data.xpos[self.root_id].copy()
    
    @property
    def orientation(self):
        """global orientation [w, x, y, z] of the car
        
        Returns:
            np.array: [w, x, y, z]
        """
        return self.data.xquat[self.root_id].copy()
    
    @property
    def rpy(self):
        """global orientation [x, y, z] of the car
        
        Returns:
            np.array: [x, y, z]
        """
        quat = self.data.xquat[self.root_id].copy()
        r = R.from_quat(quat[[1,2,3,0]])
        orientation = r.as_euler("xyz")
        return np.array(
            [
                orientation[0],
                orientation[1],
                orientation[2],
            ]
        ).copy()
        
    @property
    def lin_vel(self):
        """linear velocity [vx, vy, vz] of the car
        
        Returns:
            np.array: [vx, vy, vz]
        """
        return self.data.sensor("root_lin_vel").data.copy()
        
    @property
    def ang_vel(self):
        """angular velocity [wx, wy, wz] of the car
        
        Returns:
            np.array: [wx, wy, wz]
        """
        return self.data.sensor("root_ang_vel").data.copy()
        
    @property
    def lin_acc(self):
        """linear acceleration [ax, ay, az] of the car
        
        Returns:
            np.array: [ax, ay, az]
        """
        return self.data.sensor("root_acc").data.copy()
    
    def wheel_pos(self, i):
        return self.data.sensor(f"wheel_{i}_pos").data.copy()
    
    def wheel_quat(self, i):
        return self.data.sensor(f"wheel_{i}_quat").data.copy()
    
    def wheel_lin_vel(self, i):
        return self.data.sensor(f"wheel_{i}_lin_vel").data.copy()
    
    def wheel_ang_vel(self, i):
        return self.data.sensor(f"wheel_{i}_ang_vel").data.copy()
    
    def change_parameters(self, parameters, change = True):
        if change:
            for key, item in parameters.items():
                if key == "mass":
                    self.model.body_mass[self.root_id] = item
                elif key == "com":
                    self.model.body_ipos[self.root_id] = item
                elif key == "friction":
                    self.model.geom_friction[self.root_geom_id] = item
                    for geom_id in self.rim_geom_ids:
                        self.model.geom_friction[geom_id] = item
                elif key == "wheel_parameters":
                    radius, width, mass = item
                    for geom_id in self.rim_geom_ids:
                        self.model.geom_size[geom_id][:2] = [radius, width]
                    for body_id in self.rim_body_ids:
                        self.model.body_mass[body_id] = mass
                elif key == "wheel_base":
                    front, rear = item
                    for i, body_id in enumerate(self.knuckle_body_ids):
                        self.model.body_pos[body_id][0] = front if i < 2 else rear
                elif key == "wheel_track":
                    half_track = item/2.0
                    for i, body_id in enumerate(self.knuckle_body_ids):
                        self.model.body_pos[body_id][1] = half_track if i %2==0 else -half_track
                elif key == "max_throttle":
                    self.max_throttle = item
                elif key == "max_steer":
                    self.max_steer = item
                elif key == "steer_bias":
                    self.steer_bias = item
                elif key == "wheelbase" or "sim" or "delay":
                    pass
                else:
                    print("Invalid parameter: ", key)
        else:
            print("[WARN] Not Changing Parameters")