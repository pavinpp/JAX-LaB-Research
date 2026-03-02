"""
Stability monitoring utilities for JAX-LaB multiphase simulations.

These helpers detect and diagnose numerical instabilities that arise in the
Peng-Robinson and other non-ideal equation-of-state models, particularly at
high density ratios or near phase transition boundaries.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def stable_pseudopotential_calculation(rho: jnp.ndarray, eos_pressure: jnp.ndarray) -> dict:
    """
    Identify nodes where the Peng-Robinson EOS pseudopotential term goes
    negative (pressure shock) which causes NaN in the Shan-Chen force.

    The pseudopotential psi is defined as:
        psi = sqrt(2 * (p_EOS - rho * cs^2) / (G * cs^2))

    Under Peng-Robinson, p_EOS - rho/3 can become negative in metastable
    or spinodal regions: sqrt of a negative number → NaN.

    Parameters
    ----------
    rho : jnp.ndarray, shape (nx, ny, nz, 1)
        Macroscopic density field (keepdims form from update_macroscopic).
    eos_pressure : jnp.ndarray, shape (nx, ny, nz, 1)
        EOS pressure p(rho) returned by eos.EOS(rho_tree) for one component.

    Returns
    -------
    dict with keys:
        'inner_term'      : jnp.ndarray — value under the sqrt (negative → unstable).
        'unstable_mask'   : jnp.ndarray (bool) — True where inner_term < 0.
        'num_unstable'    : scalar — count of unstable nodes.
        'min_inner'       : scalar — most negative value (worst instability).
        'rho_at_unstable' : jnp.ndarray — density at unstable sites (NaN elsewhere).
    """
    cs2 = 1.0 / 3.0
    # Shan-Chen coupling constant g is embedded in psi; we analyse the
    # pressure-excess term directly (sign-independent of G choice).
    inner_term = eos_pressure - rho * cs2

    unstable_mask = inner_term < 0.0
    num_unstable = jnp.sum(unstable_mask)
    min_inner = jnp.min(inner_term)
    rho_at_unstable = jnp.where(unstable_mask, rho, jnp.nan)

    return {
        "inner_term": inner_term,
        "unstable_mask": unstable_mask,
        "num_unstable": num_unstable,
        "min_inner": min_inner,
        "rho_at_unstable": rho_at_unstable,
    }


def log_pseudopotential_stability(rho_tree: list, eos, step: int, fluid_mask: np.ndarray = None) -> None:
    """
    Human-readable diagnostic: print stability summary for every component.

    Uses eos.EOS(rho_tree) to obtain the equation-of-state pressure p(rho)
    and checks whether p - rho * cs^2 < 0 at any fluid node, which causes
    sqrt(negative) → NaN in the Shan-Chen pseudopotential.

    Parameters
    ----------
    rho_tree : list of jnp.ndarray, each shape (nx, ny, nz, 1)
        Density pytree from update_macroscopic.
    eos : EOS object (Peng_Robinson, VanderWaal, etc.) with an .EOS(rho_tree) method.
    step : int
        Current simulation step number (for log annotation).
    fluid_mask : np.ndarray or None
        Optional boolean mask (nx, ny, nz) to restrict analysis to fluid nodes only.
    """
    # eos.EOS() takes a pytree and returns a pytree of pressures
    p_tree = list(eos.EOS(rho_tree))

    for k, (rho, p) in enumerate(zip(rho_tree, p_tree)):
        rho_np = np.array(rho)
        p_np = np.array(p)

        if fluid_mask is not None:
            # rho has keepdims trailing dim → expand fluid_mask if needed
            mask_exp = fluid_mask[..., None] if fluid_mask.ndim == rho_np.ndim - 1 else fluid_mask
            rho_fluid = rho_np[mask_exp]
            p_fluid = p_np[mask_exp]
        else:
            rho_fluid = rho_np.ravel()
            p_fluid = p_np.ravel()

        cs2 = 1.0 / 3.0
        inner = p_fluid - rho_fluid * cs2
        n_unstable = int(np.sum(inner < 0))
        frac = n_unstable / max(len(inner), 1)

        print(
            f"  [Stability | Step {step}] Component {k}: "
            f"unstable nodes={n_unstable} ({100*frac:.2f}%), "
            f"min(p - rho*cs2)={inner.min():.4e}, "
            f"rho range=[{rho_fluid.min():.4f}, {rho_fluid.max():.4f}]"
        )
