import jax.numpy as jnp
from jax import jit
from functools import partial # <-- เพิ่ม Import นี้

@jit
def compute_permeability(mean_u: jnp.ndarray, mu: float, L: float, delta_P: float) -> jnp.ndarray:
    """
    Computes the macroscopic permeability of the porous medium using Darcy's Law.
    """
    # Guard against division by zero in the initial state or blocked flow
    delta_P_safe = jnp.where(delta_P == 0.0, 1e-8, delta_P)
    k = (mean_u * mu * L) / delta_P_safe
    return k

# บอก JAX ว่าอาร์กิวเมนต์ตัวที่ 2 (index 2 คือ calculate_equilibrium_concentration) เป็น Static Function
@partial(jit, static_argnums=(2,))
def compute_supersaturation(C: jnp.ndarray, T: jnp.ndarray, calculate_equilibrium_concentration) -> jnp.ndarray:
    """
    Computes the local supersaturation field.
    """
    C_eq = calculate_equilibrium_concentration(T)
    return jnp.maximum(C - C_eq, 0.0)