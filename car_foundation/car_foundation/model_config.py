STATE_DIM = 12  # Positiono difference: Delta x, Delta y, Delta z
                # Orientation difference: so(3) representation (3). q_{t+1} = q_t * exp(so3)
                # Linear velocity difference: Delta vx, Delta vy, Delta vz
                # Angular velocity difference: Delta wx, Delta wy, Delta wz
ACTION_DIM = 6  # Px, Py, Pz, Camber, Throttle, Steering
STATIC_DIM = 12 # Initial position (3), Actuation mask (6), wheel radius (1), wheel width (1), wheel mass (1)
HISTORY_DIM = STATE_DIM + ACTION_DIM
NUM_ENTITIES = 5  # root + 4 wheels
MODEL_DIM = 64
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
