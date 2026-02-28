import jax.numpy as jnp
from jax import jit

@jit
def calculate_equilibrium_concentration(T, T_cold=25.0, T_hot=75.0, C_cold=0.4, C_hot=1.0):
    T_clamped = jnp.clip(T, T_cold, T_hot)
    slope = (C_hot - C_cold) / (T_hot - T_cold)
    return C_cold + slope * (T_clamped - T_cold)

@jit
def compute_heterogeneous_precipitation(C, T, k_r, fluid_mask, c_vectors):
    """
    Computes precipitation exclusively at fluid-solid interfaces (heterogeneous nucleation).
    """
    C_eq = calculate_equilibrium_concentration(T)
    supersat = jnp.maximum(C - C_eq, 0.0)
    
    # Identify fluid nodes adjacent to solid nodes
    # We roll the solid mask (~fluid_mask) along the 19 lattice directions.
    # If a fluid node neighbors a solid node, the sum will be > 0.
    solid_mask = ~fluid_mask
    adjacent_solid_count = jnp.zeros_like(C, dtype=jnp.int32)
    
    for i in range(19):
        shift = tuple(c_vectors[i])
        adjacent_solid_count += jnp.roll(solid_mask, shift=shift, axis=(0,1,2)).astype(jnp.int32)
    
    # Boolean array: True if it's a fluid node AND touches at least one solid node
    is_boundary_fluid = fluid_mask & (adjacent_solid_count > 0)
    
    # Apply reaction kinetics only at the boundary
    delta_C = jnp.where(is_boundary_fluid, k_r * supersat, 0.0)
    
    return delta_C