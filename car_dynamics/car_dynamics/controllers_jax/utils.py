import jax.numpy as jnp

def void_fn(*args, **kwargs):
    return

def quat_to_yaw(q):
    w, x, y, z = q
    yaw = jnp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )
    return yaw