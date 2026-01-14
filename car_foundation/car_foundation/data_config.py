from dataclasses import dataclass

@dataclass
class MujocoDataConfig:
    STATE_DIM: int = 13 # Position, Orientation (quaternion), Linear velocity, Angular velocity
    ACTION_DIM: int = 6 # Px, Py, Pz, Camber, Throttle, Steering
    STATIC_DIM: int = 12 # Initial position (3), Actuation mask (6), wheel radius (1), wheel width (1), wheel mass (1)

    STATE_X = 0
    STATE_Y = 1
    STATE_Z = 2
    STATE_QW = 3
    STATE_QX = 4
    STATE_QY = 5
    STATE_QZ = 6
    STATE_VX = 7
    STATE_VY = 8
    STATE_VZ = 9
    STATE_WX = 10
    STATE_WY = 11
    STATE_WZ = 12

    STATE_POS = slice(0, 3)
    STATE_ORI = slice(3, 7)
    STATE_LIN_VEL = slice(7, 10)
    STATE_ANG_VEL = slice(10, 13)

    ACTION_PX = 0
    ACTION_PY = 1
    ACTION_PZ = 2
    ACTION_CAMBER = 3
    ACTION_THROTTLE = 4
    ACTION_STEERING = 5

    HISTORY_STATE_START = 0
    HISTORY_STATE_END = 13
    HISTORY_ACTION_START = 13
    HISTORY_ACTION_END = 19


@dataclass
class MujocoDiffDataConfig:
    STATE_DIM: int = 12  # Positiono difference: Delta x, Delta y, Delta z
                        # Orientation difference: so(3) representation (3). q_{t+1} = q_t * exp(so3)
                        # Linear velocity difference: Delta vx, Delta vy, Delta vz
                        # Angular velocity difference: Delta wx, Delta wy, Delta wz
    ACTION_DIM: int = 6  # Px, Py, Pz, Camber, Throttle, Steering
    STATIC_DIM: int = 12 # Initial position (3), Actuation mask (6), wheel radius (1), wheel width (1), wheel mass (1)

    STATE_DX = 0
    STATE_DY = 1
    STATE_DZ = 2
    STATE_DSO3_X = 3
    STATE_DSO3_Y = 4
    STATE_DSO3_Z = 5
    STATE_DVX = 6
    STATE_DVY = 7
    STATE_DVZ = 8
    STATE_DWX = 9
    STATE_DWY = 10
    STATE_DWZ = 11

    STATE_POS_DIFF = slice(0, 3)
    STATE_ORI_DIFF = slice(3, 6)
    STATE_LIN_VEL_DIFF = slice(6, 9)
    STATE_ANG_VEL_DIFF = slice(9, 12)