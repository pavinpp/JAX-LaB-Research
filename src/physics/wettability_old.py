import jax.numpy as jnp
from jax import jit

@jit
def compute_virtual_density(rho: jnp.ndarray, solid_mask: jnp.ndarray, fluid_mask: jnp.ndarray, theta: float, phi: float, delta_rho: float) -> jnp.ndarray:
    """
    Improved virtual density scheme for precise contact angle control on curved boundaries.
    
    Equation: 
    rho_s = phi * rho + (1 - phi) * rho_ave + delta_rho (if theta <= pi/2)
    rho_s = phi * rho + (1 - phi) * rho_ave - delta_rho (if theta > pi/2)
    
    Args:
        rho: Current macroscopic density field.
        solid_mask: Boolean array indicating solid matrix.
        fluid_mask: Boolean array indicating pore space.
        theta: Contact angle in radians.
        phi: Weighting parameter for local vs global density.
        delta_rho: Density tuning parameter for interaction strength.
        
    Returns:
        rho_new: Density field updated with virtual wall densities.
    """
    pi_over_2 = jnp.pi / 2.0
    
    # Calculate global average fluid density
    fluid_sum = jnp.sum(rho * fluid_mask)
    fluid_count = jnp.maximum(jnp.sum(fluid_mask), 1.0)
    rho_ave = fluid_sum / fluid_count
    
    # Determine thermodynamic wetting sign based on contact angle
    sign = jnp.where(theta <= pi_over_2, 1.0, -1.0)
    
    # Compute virtual density
    rho_s = (phi * rho) + ((1.0 - phi) * rho_ave) + (sign * delta_rho)
    
    # Apply the virtual density exclusively to the solid nodes
    rho_new = jnp.where(solid_mask, rho_s, rho)
    
    return rho_new