
def render_car():
    import mujoco
    import mujoco.viewer
    from car_foundation.car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR, CAR_FOUNDATION_LOG_DIR

    # from car_foundation.torch_models import DynamicsPredictor
    from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar

    env = MuJoCoCar({
        'is_render': False,
        'wheel_configs': [
            {"pos":"0.1385  0.115  0.0488",  "mask":[True, False, True, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
            {"pos":"0.1385 -0.115  0.0488",  "mask":[True, False, True, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
            {"pos":"-0.158  0.115  0.0488",  "mask":[True, False, True, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
            {"pos":"-0.158 -0.115  0.0488",  "mask":[True, False, True, False, True, True ], "radius": 0.04, "width": 0.02, "mass": 0.498952},
        ]
    })
    env.reset()

    model = env.world.model
    data = env.world.data
    print("Wheel size: ", model.geom_size[env.world.rim_geom_ids[0]])
    print("Wheel mass: ", model.body_mass[env.world.rim_body_ids[0]])
    print("Root mass: ", model.body_mass[env.world.root_id])
    print("Wheel pos: ", [model.body_pos[i] for i in env.world.knuckle_body_ids])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera('track').id
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

def load_jax_model():
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint
    from flax.training import train_state
    from orbax.checkpoint import utils as orbax_utils
    from car_foundation.jax_models import JaxDynamicsPredictor

    class TrainState(train_state.TrainState):
        rng: jax.Array

    model = JaxDynamicsPredictor(
        model_dim=128,
        output_dim=13
    )

    # 예시 차원 (저장 당시와 동일해야 함)
    BATCH = 1
    T = 100
    NUM_ENTITIES = 5
    HISTORY_DIM = 19
    STATIC_DIM = 6

    dummy_hist = jnp.ones((BATCH, T, NUM_ENTITIES, HISTORY_DIM))
    dummy_static = jnp.ones((BATCH, NUM_ENTITIES, STATIC_DIM))

    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_hist, dummy_static)
    params = variables['params']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(1e-4),
        rng=rng
    )

    MODEL_SAVE_PATH = os.path.join(CAR_FOUNDATION_MODEL_DIR, '2026-01-08T15:48:33.113-model_checkpoint')

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    manager = orbax.checkpoint.CheckpointManager(
        MODEL_SAVE_PATH,
        checkpointer,
        orbax.checkpoint.CheckpointManagerOptions()
    )

    step = manager.latest_step()
    print("Restoring checkpoint at step:", step)

    restore_target = {
        'model': state,
        'config': {
            'target_scale': 100.0,
            'dims': (HISTORY_DIM, STATIC_DIM)
        }
    }

    ckpt = manager.restore(
        step,
        restore_target
    )

    hist_jax = jnp.ones((1, 10, NUM_ENTITIES, HISTORY_DIM))
    static_jax = jnp.ones((1, NUM_ENTITIES, STATIC_DIM))

    key = jax.random.PRNGKey(123)
    key, key2 = jax.random.split(key, 2)

    pred = state.apply_fn(
        {'params': state.params},
        hist_jax,
        static_jax,
        rngs = key2,
    )

render_car()