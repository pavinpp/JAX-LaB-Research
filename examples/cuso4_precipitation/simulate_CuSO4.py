import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit

import sys
import os
sys.path.append(os.path.abspath("../../"))

from src.lattice import LatticeD3Q19
from src.physics.crystallization import compute_heterogeneous_precipitation

def parse_ui_args():
    parser = argparse.ArgumentParser(description="JAX-LaB CuSO4 Crystallization")
    parser.add_argument("--geom", type=str, default="geometry_mask.npy")
    parser.add_argument("--axis", type=str, choices=['X', 'Y', 'Z'], default='X')
    parser.add_argument("--flow_rate", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--inject_size", type=int, default=60)
    return parser.parse_args()

def run_simulation():
    args = parse_ui_args()
    
    print(f"Loading geometry and cropping to {args.inject_size}^3...")
    mask_np_full = np.load(args.geom).astype(bool)
    
    # Crop geometry exactly as before
    c_size = args.inject_size
    half_c = c_size // 2
    cx_o, cy_o, cz_o = mask_np_full.shape[0]//2, mask_np_full.shape[1]//2, mask_np_full.shape[2]//2
    xs, xe = cx_o - half_c, cx_o + half_c
    ys, ye = cy_o - half_c, cy_o + half_c
    zs, ze = cz_o - half_c, cz_o + half_c
    
    mask = jnp.array(mask_np_full[xs:xe, ys:ye, zs:ze])
    nx, ny, nz = mask.shape
    
    lattice = LatticeD3Q19()
    # Define c_int as a pure Python list of ints so it acts as a static constant during JIT!
    c_int = np.array(lattice.c, dtype=int).T.tolist()   
    c = jnp.array(lattice.c, dtype=jnp.float32).T       # JAX array for math
    w = jnp.array(lattice.w, dtype=jnp.float32)
    
    # Physics Parameters
    u_lb = 0.01  # Lattice velocity
    # Ensure thermal diffusion is MUCH faster than solute diffusion
    # tau closer to 0.5 = higher diffusivity/viscosity in LBM units
    tau_f = 1.0   # Hydrodynamics
    tau_t = 0.55  # Thermal (Fast heat dissipation into the cold beads)
    tau_c = 0.95  # Solute (Slow mass diffusion, keeping C high)
    
    omega_f, omega_t, omega_c = 1.0/tau_f, 1.0/tau_t, 1.0/tau_c
    k_r = 0.15    # Reaction rate
    
    # Target Inlet Conditions
    T_hot = 75.0
    T_cold = 25.0
    C_inlet = 1.0  # Saturated at 75C
    target_u = jnp.zeros(3).at[0].set(u_lb) # Assuming X-axis for simplicity
    
    # Initialize Macroscopic Fields
    rho = jnp.ones(mask.shape, dtype=jnp.float32)
    u = jnp.zeros(mask.shape + (3,), dtype=jnp.float32)
    T = jnp.ones(mask.shape, dtype=jnp.float32) * T_cold
    C = jnp.zeros(mask.shape, dtype=jnp.float32)
    solid_fraction = jnp.zeros(mask.shape, dtype=jnp.float32)

    @jit
    def calc_equilibrium(phi, u):
        cu = jnp.dot(u, c.T)
        usqr = jnp.sum(u**2, axis=-1, keepdims=True)
        return phi[..., None] * w * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)
    
    # Initialize populations
    f = calc_equilibrium(rho, u)
    g = calc_equilibrium(T, u)
    h = calc_equilibrium(C, u)
    
    def shift_no_wrap(a, sx, sy, sz):
        out = jnp.zeros_like(a)
        xs_src = slice(max(-sx, 0), a.shape[0] - max(sx, 0))
        ys_src = slice(max(-sy, 0), a.shape[1] - max(sy, 0))
        zs_src = slice(max(-sz, 0), a.shape[2] - max(sz, 0))
        xs_dst = slice(max(sx, 0), a.shape[0] - max(-sx, 0))
        ys_dst = slice(max(sy, 0), a.shape[1] - max(-sy, 0))
        zs_dst = slice(max(sz, 0), a.shape[2] - max(-sz, 0))
        return out.at[xs_dst, ys_dst, zs_dst].set(a[xs_src, ys_src, zs_src])

    @jit
    def lbm_step(state, _):
        f, g, h, solid_frac = state
        
        # 1. Macroscopic variables
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, c) / rho[..., None]
        T = jnp.sum(g, axis=-1)
        C = jnp.sum(h, axis=-1)
        
        # 2. Impose UI Inlet manually for stability
        if args.axis == 'X':
            u = u.at[0, :, :, :].set(target_u)
            rho = rho.at[0, :, :].set(1.0)
            T = T.at[0, :, :].set(T_hot)
            C = C.at[0, :, :].set(C_inlet)
            
        # 3. Collision
        f_post = f - omega_f * (f - calc_equilibrium(rho, u))
        g_post = g - omega_t * (g - calc_equilibrium(T, u))
        h_post = h - omega_c * (h - calc_equilibrium(C, u))
        
        # 4. Kinetics: Heterogeneous Precipitation at Solid Interfaces
        effective_fluid_mask = mask & (solid_frac < 0.5)

        # Mark side planes as solid walls so they participate in bounce-back and interface detection
        wall_mask = jnp.zeros_like(effective_fluid_mask, dtype=bool)
        if args.axis == 'X':
            wall_mask = wall_mask.at[:, 0, :].set(True)
            wall_mask = wall_mask.at[:, -1, :].set(True)
            wall_mask = wall_mask.at[:, :, 0].set(True)
            wall_mask = wall_mask.at[:, :, -1].set(True)
        elif args.axis == 'Y':
            wall_mask = wall_mask.at[0, :, :].set(True)
            wall_mask = wall_mask.at[-1, :, :].set(True)
            wall_mask = wall_mask.at[:, :, 0].set(True)
            wall_mask = wall_mask.at[:, :, -1].set(True)
        elif args.axis == 'Z':
            wall_mask = wall_mask.at[0, :, :].set(True)
            wall_mask = wall_mask.at[-1, :, :].set(True)
            wall_mask = wall_mask.at[:, 0, :].set(True)
            wall_mask = wall_mask.at[:, -1, :].set(True)
            
        effective_fluid_mask_bc = effective_fluid_mask & (~wall_mask)

        delta_C = compute_heterogeneous_precipitation(C, T, k_r, effective_fluid_mask_bc, c_int)
        
        h_post = h_post - w * delta_C[..., None]
        solid_frac = solid_frac + delta_C
        
        # 5. Streaming (Non-Periodic with Robust Half-Way Bounce-Back)
        f_str = jnp.zeros_like(f)
        g_str = jnp.zeros_like(g)
        h_str = jnp.zeros_like(h)
        
        solid_mask = ~effective_fluid_mask_bc
        domain_ones = jnp.ones_like(mask, dtype=bool)
        opp = jnp.array(lattice.opp_indices)
        
        for i in range(19):
            sx, sy, sz = c_int[i][0], c_int[i][1], c_int[i][2]
            f_str = f_str.at[..., i].set(shift_no_wrap(f_post[..., i], sx, sy, sz))
            
            # Shift without wrap
            f_streamed_i = shift_no_wrap(f_post[..., i], sx, sy, sz)
            g_streamed_i = shift_no_wrap(g_post[..., i], sx, sy, sz)
            h_streamed_i = shift_no_wrap(h_post[..., i], sx, sy, sz)
            
            # Detect if the source of this shifted population was a solid node OR outside the domain bounds
            source_was_solid = shift_no_wrap(solid_mask, sx, sy, sz)
            source_was_outside = ~shift_no_wrap(domain_ones, sx, sy, sz)
            is_invalid_source = source_was_solid | source_was_outside
            
            # Apply Half-Way Bounce-Back dynamically at fluid nodes
            f_str = f_str.at[..., i].set(jnp.where(is_invalid_source, f_post[..., opp[i]], f_streamed_i))
            g_str = g_str.at[..., i].set(jnp.where(is_invalid_source, g_post[..., opp[i]], g_streamed_i))
            h_str = h_str.at[..., i].set(jnp.where(is_invalid_source, h_post[..., opp[i]], h_streamed_i))
        
        # 6. Inlet and Outlet Boundary Conditions
        if args.axis == 'X':
            # Hard Dirichlet Equilibrium at Inlet (X=0)
            u_in = jnp.zeros_like(u[0]).at[..., 0].set(u_lb)
            f_str = f_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(rho[0]), u_in))
            g_str = g_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(T[0]) * T_hot, u_in))
            h_str = h_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(C[0]) * C_inlet, u_in))
            
            # Outflow Extrapolation at Exit (X=-1)
            f_str = f_str.at[-1, :, :, :].set(f_str[-2, :, :, :])
            g_str = g_str.at[-1, :, :, :].set(g_str[-2, :, :, :])
            h_str = h_str.at[-1, :, :, :].set(h_str[-2, :, :, :])
            
        elif args.axis == 'Y':
            u_in = jnp.zeros_like(u[:, 0]).at[..., 1].set(u_lb)
            f_str = f_str.at[:, 0, :, :].set(calc_equilibrium(jnp.ones_like(rho[:, 0]), u_in))
            g_str = g_str.at[:, 0, :, :].set(calc_equilibrium(jnp.ones_like(T[:, 0]) * T_hot, u_in))
            h_str = h_str.at[:, 0, :, :].set(calc_equilibrium(jnp.ones_like(C[:, 0]) * C_inlet, u_in))
            
            f_str = f_str.at[:, -1, :, :].set(f_str[:, -2, :, :])
            g_str = g_str.at[:, -1, :, :].set(g_str[:, -2, :, :])
            h_str = h_str.at[:, -1, :, :].set(h_str[:, -2, :, :])
            
        elif args.axis == 'Z':
            u_in = jnp.zeros_like(u[:, :, 0]).at[..., 2].set(u_lb)
            f_str = f_str.at[:, :, 0, :].set(calc_equilibrium(jnp.ones_like(rho[:, :, 0]), u_in))
            g_str = g_str.at[:, :, 0, :].set(calc_equilibrium(jnp.ones_like(T[:, :, 0]) * T_hot, u_in))
            h_str = h_str.at[:, :, 0, :].set(calc_equilibrium(jnp.ones_like(C[:, :, 0]) * C_inlet, u_in))
            
            f_str = f_str.at[:, :, -1, :].set(f_str[:, :, -2, :])
            g_str = g_str.at[:, :, -1, :].set(g_str[:, :, -2, :])
            h_str = h_str.at[:, :, -1, :].set(h_str[:, :, -2, :])

        return (f_str, g_str, h_str, solid_frac), None

    # --- Chunked Execution with Temporal Tracking ---
    chunk_size = 500
    num_chunks = args.steps // chunk_size
    
    @jit(static_argnums=(1,))
    def run_chunk(state_in, steps):
        state_out, _ = jax.lax.scan(lbm_step, state_in, jnp.arange(steps))
        return state_out

    print(f"Running Reactive {args.steps} LBM steps...")
    state = (f, g, h, solid_fraction)
    
    # Initialize history lists for our graphs
    history_steps = []
    history_T_mean = []
    history_C_mean = []
    history_supersat_max = []
    history_crystal_vol = []
    
    for i in range(num_chunks):
        state = run_chunk(state, chunk_size)
        state[0].block_until_ready()
        
        # --- Extract Macro Fields for Diagnostics ---
        f_curr, g_curr, h_curr, solid_curr = state
        T_curr = jnp.sum(g_curr, axis=-1)
        C_curr = jnp.sum(h_curr, axis=-1)
        
        # Calculate Equilibrium and Supersaturation manually for logging
        T_clamped = jnp.clip(T_curr, 25.0, 75.0)
        # Using the same linear slope from crystallization.py
        C_eq = 0.4 + (1.0 - 0.4) / (75.0 - 25.0) * (T_clamped - 25.0)
        supersat = jnp.maximum(C_curr - C_eq, 0.0)
        
        # Only measure fluid voxels (ignore the solid glass beads)
        fluid_T = T_curr[mask]
        fluid_C = C_curr[mask]
        fluid_supersat = supersat[mask]
        
        # Append to history
        current_step = (i+1) * chunk_size
        history_steps.append(current_step)
        history_T_mean.append(float(jnp.mean(fluid_T)))
        history_C_mean.append(float(jnp.mean(fluid_C)))
        history_supersat_max.append(float(jnp.max(fluid_supersat)))
        
        current_solid = jnp.sum(solid_curr)
        history_crystal_vol.append(float(current_solid))
        
        print(f"Step {current_step}/{args.steps} | Max Supersat: {history_supersat_max[-1]:.4f} | Crystal Vol: {current_solid:.2f}")

    # Return the history lists along with the final state
    return state, mask, (history_steps, history_T_mean, history_C_mean, history_supersat_max, history_crystal_vol)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Unpack the simulation results and the new history tuple
    final_state, mask_np, history = run_simulation()
    
    f_final, g_final, h_final, solid_final = final_state
    history_steps, hist_T, hist_C, hist_supersat, hist_vol = history
    
    # Extract final macro fields for spatial mapping
    T_final = np.array(jnp.sum(g_final, axis=-1))
    C_final = np.array(jnp.sum(h_final, axis=-1))
    crystal_map = np.array(solid_final)
    
    mid_z = mask_np.shape[2] // 2
    
    # ==========================================================
    # --- PLOT 1: Spatial Reaction Maps (The 3-Pane View) ---
    # ==========================================================
    print("\nGenerating Spatial Reaction Maps...")
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Temperature Map
    T_slice = T_final[:, :, mid_z].astype(float)
    T_slice[~mask_np[:, :, mid_z]] = np.nan
    im0 = axes[0].imshow(T_slice.T, cmap='inferno', origin='lower', vmin=25, vmax=75)
    axes[0].set_title('Temperature Field (°C)')
    fig1.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Concentration Map
    C_slice = C_final[:, :, mid_z].astype(float)
    C_slice[~mask_np[:, :, mid_z]] = np.nan
    im1 = axes[1].imshow(C_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1.0)
    axes[1].set_title('CuSO4 Aqueous Concentration')
    fig1.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Solid Crystal Precipitation Map
    solid_slice = crystal_map[:, :, mid_z].astype(float)
    solid_slice[~mask_np[:, :, mid_z]] = np.nan
    im2 = axes[2].imshow(solid_slice.T, cmap='cool', origin='lower')
    axes[2].set_title('Precipitated Crystal Volume (Solid Fraction)')
    fig1.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig1.suptitle("Reactive Transport: Cooling-Induced CuSO4 Crystallization", fontsize=16)
    fig1.tight_layout()
    fig1.savefig("cuso4_reaction_maps.png", dpi=300, bbox_inches='tight')
    print("Exported spatial results to 'cuso4_reaction_maps.png'!")

    # ==========================================================
    # --- PLOT 2: Temporal Diagnostic Dashboard ---
    # ==========================================================
    print("\nGenerating Time-Series Diagnostic Dashboard...")
    fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-Left: Mean Temperature
    axs[0, 0].plot(history_steps, hist_T, 'r-', linewidth=2)
    axs[0, 0].set_title("Mean Fluid Temperature over Time")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid(True)
    
    # Top-Right: Mean Concentration
    axs[0, 1].plot(history_steps, hist_C, 'g-', linewidth=2)
    axs[0, 1].set_title("Mean CuSO4 Concentration over Time")
    axs[0, 1].set_ylabel("Concentration (Normalized)")
    axs[0, 1].grid(True)
    
    # Bottom-Left: Maximum Supersaturation
    axs[1, 0].plot(history_steps, hist_supersat, 'm-', linewidth=2)
    axs[1, 0].set_title("Maximum Supersaturation (Nucleation Trigger)")
    axs[1, 0].set_xlabel("Time Steps")
    axs[1, 0].set_ylabel("Max(C - C_eq)")
    axs[1, 0].grid(True)
    
    # Bottom-Right: Total Crystal Volume
    axs[1, 1].plot(history_steps, hist_vol, 'b-', linewidth=2)
    axs[1, 1].set_title("Total Precipitated Crystal Volume")
    axs[1, 1].set_xlabel("Time Steps")
    axs[1, 1].set_ylabel("Volume (Lattice Units)")
    axs[1, 1].grid(True)
    
    fig2.suptitle("LBM Reactive Transport: Temporal Evolution", fontsize=16)
    fig2.tight_layout()
    fig2.savefig("cuso4_temporal_dashboard.png", dpi=300, bbox_inches='tight')
    print("Exported time-series to 'cuso4_temporal_dashboard.png'!")