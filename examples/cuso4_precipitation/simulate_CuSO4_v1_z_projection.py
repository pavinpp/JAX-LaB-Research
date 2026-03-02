import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
import csv
import os
import sys
import scipy.ndimage as ndimage
import pyvista as pv

sys.path.append(os.path.abspath("../../"))

from src.lattice import LatticeD3Q19
from src.physics.crystallization import compute_heterogeneous_precipitation, calculate_equilibrium_concentration
from src.physics.porous_media import compute_permeability, compute_supersaturation

def parse_ui_args():
    parser = argparse.ArgumentParser(description="JAX-LaB CuSO4 Crystallization")
    parser.add_argument("--geom", type=str, default="geometry_mask.npy")
    parser.add_argument("--axis", type=str, choices=['X', 'Y', 'Z'], default='X')
    parser.add_argument("--flow_rate", type=float, default=1.0, help="Flow rate in mL/hr")
    parser.add_argument("--dx_um", type=float, default=20.0, help="Voxel size in micrometers (um)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--inject_size", type=int, default=60)
    return parser.parse_args()

def save_vti_file(filename, array, name, is_vector=False):
    if is_vector:
        dimensions = array.shape[:-1]
        grid = pv.ImageData(dimensions=dimensions)
        grid.point_data[name] = array.reshape(-1, 3, order="F")
    else:
        dimensions = array.shape
        grid = pv.ImageData(dimensions=dimensions)
        grid.point_data[name] = array.flatten(order="F")
    grid.save(filename)

@jit
def calculate_tau_f(T_celsius, tau_ref=1.0):
    temp_points = jnp.array([25.0, 35.0, 45.0, 55.0, 65.0, 75.0])
    viscosity_points = jnp.array([1.35, 1.08, 0.89, 0.74, 0.63, 0.55])
    mu_T = jnp.interp(T_celsius, temp_points, viscosity_points)
    mu_ref = 1.35 
    tau_f = 0.5 + (tau_ref - 0.5) * (mu_T / mu_ref)
    return tau_f

def run_simulation():
    args = parse_ui_args()
    
    print(f"Loading geometry and cropping to {args.inject_size}^3...")
    mask_np_full = np.load(args.geom).astype(bool)
    
    c_size = args.inject_size
    half_c = c_size // 2
    cx_o, cy_o, cz_o = mask_np_full.shape[0]//2, mask_np_full.shape[1]//2, mask_np_full.shape[2]//2
    xs, xe = cx_o - half_c, cx_o + half_c
    ys, ye = cy_o - half_c, cy_o + half_c
    zs, ze = cz_o - half_c, cz_o + half_c
    
    mask = jnp.array(mask_np_full[xs:xe, ys:ye, zs:ze])
    
    lattice = LatticeD3Q19()
    c_int = np.array(lattice.c, dtype=int).T.tolist()   
    c = jnp.array(lattice.c, dtype=jnp.float32).T       
    w = jnp.array(lattice.w, dtype=jnp.float32)
    c_np = np.array(lattice.c, dtype=np.float32).T 
    
    dx_m = args.dx_um * 1e-6    
    dx_mm = args.dx_um * 1e-3   
    nu_phys = 1e-6 
    
    tau_f_ref = 1.0   
    tau_t = 0.55  
    tau_c = 0.95  
    
    nu_lb = (tau_f_ref - 0.5) / 3.0
    dt_s = (nu_lb * (dx_m ** 2)) / nu_phys 
    
    Q_m3s = args.flow_rate / 3.6e9  
    cross_section_area_m2 = (mask.shape[1] * mask.shape[2]) * (dx_m ** 2)
    u_phys_inlet = Q_m3s / cross_section_area_m2 
    u_lb = u_phys_inlet * (dt_s / dx_m)          
    
    vol_scale_mm3 = dx_mm ** 3
    k_scale_m2 = dx_m ** 2
    k_scale_darcy = k_scale_m2 / 0.9869233e-12  
    u_scale_mms = (dx_mm / dt_s)                
    
    print("\n--- Physical Scales Confirmed ---")
    print(f"  Voxel Size: {args.dx_um} um")
    print(f"  Time Step (dt): {dt_s:.2e} s")
    print(f"  Inlet Velocity (Target): {u_phys_inlet*1000:.2f} mm/s (LBM: {u_lb:.4f})")
    print("---------------------------------\n")

    omega_t, omega_c = 1.0/tau_t, 1.0/tau_c
    k_r = 0.15    
    T_hot, T_cold, C_inlet = 75.0, 25.0, 1.0
    
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
    def lbm_step(state, step_idx):
        f, g, h, solid_frac = state
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, c) / rho[..., None]
        T = jnp.sum(g, axis=-1)
        C = jnp.sum(h, axis=-1)
        
        ramp_factor = jnp.clip(step_idx / 500.0, 0.0, 1.0)
        current_u_lb = u_lb * ramp_factor
        target_u_dynamic = jnp.zeros(3).at[0].set(current_u_lb)
        
        if args.axis == 'X':
            u = u.at[0, :, :, :].set(target_u_dynamic)
            rho = rho.at[0, :, :].set(1.0)
            T = T.at[0, :, :].set(T_hot)
            C = C.at[0, :, :].set(C_inlet)
            
        tau_f_local = calculate_tau_f(T, tau_ref=tau_f_ref)
        omega_f_local = 1.0 / tau_f_local
        
        f_post = f - omega_f_local[..., None] * (f - calc_equilibrium(rho, u))
        g_post = g - omega_t * (g - calc_equilibrium(T, u))
        h_post = h - omega_c * (h - calc_equilibrium(C, u))
        
        effective_fluid_mask = mask & (solid_frac < 0.5)
        wall_mask = jnp.zeros_like(effective_fluid_mask, dtype=bool)
        if args.axis == 'X':
            wall_mask = wall_mask.at[:, 0, :].set(True)
            wall_mask = wall_mask.at[:, -1, :].set(True)
            wall_mask = wall_mask.at[:, :, 0].set(True)
            wall_mask = wall_mask.at[:, :, -1].set(True)
            
        effective_fluid_mask_bc = effective_fluid_mask & (~wall_mask)
        delta_C = compute_heterogeneous_precipitation(C, T, k_r, effective_fluid_mask_bc, c_int)
        
        h_post = h_post - w * delta_C[..., None]
        solid_frac = solid_frac + delta_C
        
        f_str, g_str, h_str = jnp.zeros_like(f), jnp.zeros_like(g), jnp.zeros_like(h)
        solid_mask = ~effective_fluid_mask_bc
        domain_ones = jnp.ones_like(mask, dtype=bool)
        opp = jnp.array(lattice.opp_indices)
        
        for i in range(19):
            sx, sy, sz = c_int[i][0], c_int[i][1], c_int[i][2]
            f_streamed_i = shift_no_wrap(f_post[..., i], sx, sy, sz)
            g_streamed_i = shift_no_wrap(g_post[..., i], sx, sy, sz)
            h_streamed_i = shift_no_wrap(h_post[..., i], sx, sy, sz)
            
            is_invalid = shift_no_wrap(solid_mask, sx, sy, sz) | ~shift_no_wrap(domain_ones, sx, sy, sz)
            
            f_str = f_str.at[..., i].set(jnp.where(is_invalid, f_post[..., opp[i]], f_streamed_i))
            g_str = g_str.at[..., i].set(jnp.where(is_invalid, g_post[..., opp[i]], g_streamed_i))
            h_str = h_str.at[..., i].set(jnp.where(is_invalid, h_post[..., opp[i]], h_streamed_i))
        
        if args.axis == 'X':
            u_in = jnp.zeros_like(u[0]).at[..., 0].set(current_u_lb)
            f_str = f_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(rho[0]), u_in))
            g_str = g_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(T[0]) * T_hot, u_in))
            h_str = h_str.at[0, :, :, :].set(calc_equilibrium(jnp.ones_like(C[0]) * C_inlet, u_in))
            
            f_str = f_str.at[-1, :, :, :].set(f_str[-2, :, :, :])
            g_str = g_str.at[-1, :, :, :].set(g_str[-2, :, :, :])
            h_str = h_str.at[-1, :, :, :].set(h_str[-2, :, :, :])

        return (f_str, g_str, h_str, solid_frac), None

    chunk_size = 500
    num_chunks = args.steps // chunk_size
    
    @jit(static_argnums=(1,))
    def run_chunk(state_in, steps, chunk_start):
        step_indices = jnp.arange(steps) + chunk_start
        state_out, _ = jax.lax.scan(lbm_step, state_in, step_indices)
        return state_out

    os.makedirs("outputs/vti", exist_ok=True)
    os.makedirs("outputs/analytics", exist_ok=True)
    
    with open("outputs/global_kinetics.csv", "w", newline="") as f1, \
         open("outputs/object_analysis.csv", "w", newline="") as f2, \
         open("outputs/pore_clogging_stats.csv", "w", newline="") as f3:
        csv.writer(f1).writerow(["Step", "Time_s", "Total_Solid_Volume_mm3", "Porosity", "Global_Permeability_Darcy", "Avg_Temperature"])
        csv.writer(f2).writerow(["Step", "Number_of_Crystals", "Avg_Crystal_Size", "Max_Crystal_Size", "Surface_Area"])
        csv.writer(f3).writerow(["Step", "Min_Throat_Size", "Tortuosity_Index"])

    domain_length = float(args.inject_size)
    D_solute = (1.0/3.0) * (tau_c - 0.5)

    print(f"Running Reactive {args.steps} LBM steps with comprehensive I/O...")
    state = (f, g, h, solid_fraction)
    mask_cpu = np.array(mask)
    mid_x, mid_y, mid_z = mask_cpu.shape[0]//2, mask_cpu.shape[1]//2, mask_cpu.shape[2]//2
    
    vel_mag_t0 = None
    vel_mag_tfinal = None
    pe_da_data = [] 
    maps_data = {} 
    
    for i in range(num_chunks + 1):
        if i > 0:
            chunk_start_step = (i - 1) * chunk_size
            state = run_chunk(state, chunk_size, chunk_start_step)
            
        state[0].block_until_ready()
        current_step = i * chunk_size
        
        if current_step == 0 or current_step % 500 == 0:
            f_np, g_np, h_np, solid_np = [np.array(x) for x in state]
            T_np = np.sum(g_np, axis=-1)
            C_np = np.sum(h_np, axis=-1)
            rho_np = np.sum(f_np, axis=-1)
            u_np = np.dot(f_np, c_np) / rho_np[..., None]
            
            binary_precipitate = np.where(solid_np > 0.1, 1.0, 0.0).astype(np.float32)
            fluid_mask_current = mask_cpu & (solid_np < 0.5)
            supersat_map = np.array(compute_supersaturation(jnp.array(C_np), jnp.array(T_np), calculate_equilibrium_concentration))
            u_mag = np.linalg.norm(u_np, axis=-1)
            
            if current_step == 0:
                vel_mag_t0 = u_mag[fluid_mask_current]

            # --- เพิ่ม Z_proj (Sum ตามแกน Z) เพื่อทำ Top-Down Projection ---
            maps_data[current_step] = {
                'XY': {'T': T_np[:, :, mid_z].copy(), 'C': C_np[:, :, mid_z].copy(), 'solid': solid_np[:, :, mid_z].copy()},
                'XZ': {'T': T_np[:, mid_y, :].copy(), 'C': C_np[:, mid_y, :].copy(), 'solid': solid_np[:, mid_y, :].copy()},
                'YZ': {'T': T_np[mid_x, :, :].copy(), 'C': C_np[mid_x, :, :].copy(), 'solid': solid_np[mid_x, :, :].copy()},
                'Z_proj': {'solid_sum': np.sum(solid_np, axis=2).copy()}
            }

            save_vti_file(f"outputs/vti/precipitate_growth_t{current_step}.vti", binary_precipitate, "CuSO4_Solid")
            save_vti_file(f"outputs/vti/velocity_evolution_t{current_step}.vti", u_np, "Velocity", is_vector=True)
            save_vti_file(f"outputs/vti/supersaturation_map_t{current_step}.vti", supersat_map, "Supersaturation")

            delta_P = (np.mean(rho_np[0, :, :]) - np.mean(rho_np[-1, :, :])) / 3.0
            mean_u = np.mean(u_np[..., 0]) 
            
            avg_T = np.mean(T_np)
            tau_f_avg = float(calculate_tau_f(jnp.array(avg_T), tau_ref=tau_f_ref))
            mu_fluid_avg = (tau_f_avg - 0.5) / 3.0
            
            k_raw = float(np.array(compute_permeability(mean_u, mu_fluid_avg, domain_length, delta_P)))
            k_perm_darcy = abs(k_raw) * k_scale_darcy
            
            current_solid_vol_mm3 = np.sum(solid_np) * vol_scale_mm3
            time_s = current_step * dt_s 
            
            total_voxels = mask_cpu.size         
            initial_fluid_voxels = np.sum(mask_cpu) 
            porosity = (initial_fluid_voxels - np.sum(solid_np)) / total_voxels

            labeled_array, num_features = ndimage.label(binary_precipitate > 0)
            avg_size, max_size = 0, 0
            if num_features > 0:
                sizes = np.bincount(labeled_array.ravel())[1:]
                avg_size = np.mean(sizes)
                max_size = np.max(sizes)
            
            solid_dilated = ndimage.binary_dilation(binary_precipitate > 0)
            surface_area = np.sum(solid_dilated & fluid_mask_current)

            u_x = u_np[..., 0]
            tortuosity = np.sum(u_mag[fluid_mask_current]) / (np.sum(u_x[fluid_mask_current]) + 1e-8)
            
            dt = ndimage.distance_transform_edt(fluid_mask_current)
            max_r_per_slice = [np.max(dt[x, :, :]) for x in range(dt.shape[0])]
            min_throat = np.min(max_r_per_slice)

            max_supersat = np.max(supersat_map[fluid_mask_current]) if np.any(fluid_mask_current) else 0.0

            with open("outputs/global_kinetics.csv", "a", newline="") as f1, \
                 open("outputs/object_analysis.csv", "a", newline="") as f2, \
                 open("outputs/pore_clogging_stats.csv", "a", newline="") as f3:
                csv.writer(f1).writerow([current_step, time_s, current_solid_vol_mm3, porosity, k_perm_darcy, avg_T])
                csv.writer(f2).writerow([current_step, num_features, avg_size, max_size, surface_area])
                csv.writer(f3).writerow([current_step, min_throat, tortuosity])

            print(f"Step {current_step}/{args.steps} | Time: {time_s:.2f} s | Porosity: {porosity:.4f} | Crystals: {num_features} | k: {k_perm_darcy:.2e} Darcy | Max Supersat: {max_supersat:.4f} | Crystal Vol: {current_solid_vol_mm3:.2e} mm3")

    vel_mag_tfinal = u_mag[fluid_mask_current]
    
    u_mag_safe = np.where(u_mag == 0, 1e-8, u_mag)
    L_ref = 1.0 
    Pe_map = (u_mag * L_ref) / D_solute
    Da_map = (k_r * L_ref) / u_mag_safe
    pe_da_data = (Pe_map[fluid_mask_current], Da_map[fluid_mask_current])

    return (vel_mag_t0, vel_mag_tfinal, pe_da_data, maps_data, mask_cpu, u_scale_mms)

def generate_reaction_maps(maps_data, mask_np):
    print("\nGenerating Spatial Reaction Maps (XY, XZ, YZ, and Z-Projection)...")
    steps_saved = sorted(list(maps_data.keys()))
    
    if len(steps_saved) >= 3:
        steps_to_plot = [steps_saved[0], steps_saved[len(steps_saved)//2], steps_saved[-1]]
    else:
        steps_to_plot = steps_saved
        
    mid_x, mid_y, mid_z = mask_np.shape[0]//2, mask_np.shape[1]//2, mask_np.shape[2]//2
    
    # ---------------------------------------------------------
    # 1. พล็อต Slices แบบปกติ (XY, XZ, YZ)
    # ---------------------------------------------------------
    planes = {
        'XY': {'mask': mask_np[:, :, mid_z], 'title': 'X-Y Cross Section (Mid-Z)'},
        'XZ': {'mask': mask_np[:, mid_y, :], 'title': 'X-Z Cross Section (Mid-Y)'},
        'YZ': {'mask': mask_np[mid_x, :, :], 'title': 'Y-Z Cross Section (Mid-X)'}
    }
    
    for plane_name, plane_info in planes.items():
        fig, axes = plt.subplots(len(steps_to_plot), 3, figsize=(18, 5 * len(steps_to_plot)))
        if len(steps_to_plot) == 1: axes = np.expand_dims(axes, axis=0) 
        mask_slice = plane_info['mask']
        
        for row_idx, step in enumerate(steps_to_plot):
            T_slice = maps_data[step][plane_name]['T'].astype(float)
            C_slice = maps_data[step][plane_name]['C'].astype(float)
            solid_slice = maps_data[step][plane_name]['solid'].astype(float)
            
            T_slice[~mask_slice] = np.nan
            C_slice[~mask_slice] = np.nan
            solid_slice[~mask_slice] = np.nan
            
            im0 = axes[row_idx, 0].imshow(T_slice.T, cmap='inferno', origin='lower', vmin=25, vmax=75)
            axes[row_idx, 0].set_title(f'Step {step}: Temp (°C)')
            fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
            
            im1 = axes[row_idx, 1].imshow(C_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1.0)
            axes[row_idx, 1].set_title(f'Step {step}: CuSO4 Conc.')
            fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[row_idx, 2].imshow(solid_slice.T, cmap='cool', origin='lower')
            axes[row_idx, 2].set_title(f'Step {step}: Crystal Vol')
            fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
            
        fig.suptitle(f"Reactive Transport Evolution: {plane_info['title']}", fontsize=20)
        fig.tight_layout()
        fig.savefig(f"outputs/analytics/cuso4_reaction_maps_{plane_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # 2. พล็อต Z-Projection (Top-Down Sum of Solid) ใหม่ล่าสุด!
    # ---------------------------------------------------------
    fig_proj, axes_proj = plt.subplots(1, len(steps_to_plot), figsize=(6 * len(steps_to_plot), 5))
    if len(steps_to_plot) == 1: axes_proj = [axes_proj]

    # คำนวณหาช่องว่างรูพรุนทั้งหมดในแนวแกน Z เพื่อใช้บังส่วนที่เป็นก้อนหินทึบตัน
    pore_depth = np.sum(mask_np, axis=2).astype(float)
    pore_depth[pore_depth == 0] = np.nan 

    for col_idx, step in enumerate(steps_to_plot):
        solid_sum = maps_data[step]['Z_proj']['solid_sum'].astype(float)
        
        # ถ้าระนาบ Z ตรงนั้นไม่มีช่องว่างเลย (เป็นหินล้วน) ให้ซ่อนสีไป (กลายเป็นพื้นหลังขาว/เทา)
        solid_sum[np.isnan(pore_depth)] = np.nan 

        # พล็อตค่า Sum ลงไป (ใช้สี magma เพื่อเน้นจุดที่มีคริสตัลทับซ้อนกันหนาแน่น)
        im = axes_proj[col_idx].imshow(solid_sum.T, cmap='magma', origin='lower')
        axes_proj[col_idx].set_title(f'Step {step}: Total Crystal Depth')
        fig_proj.colorbar(im, ax=axes_proj[col_idx], fraction=0.046, pad=0.04)

    fig_proj.suptitle("Z-Projection (Top-down Sum of Crystal Volume)", fontsize=20)
    fig_proj.tight_layout()
    fig_proj.savefig("outputs/analytics/cuso4_reaction_maps_Z_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved Z-Projection Map")

def generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms):
    print("Generating Analytical PNG Plots (Physical Units)...")
    os.makedirs("outputs/analytics", exist_ok=True)
    
    kinetics = np.genfromtxt("outputs/global_kinetics.csv", delimiter=',', skip_header=1)
    objects = np.genfromtxt("outputs/object_analysis.csv", delimiter=',', skip_header=1)
    
    # 1. Permeability Reduction 
    if kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        permeability = kinetics[:, 4]
        
        plt.figure(figsize=(8, 6))
        k_mag = np.abs(permeability) 
        valid_idx = k_mag > 1e-10  
        
        if np.any(valid_idx):
            plt.plot(time_s[valid_idx], k_mag[valid_idx], 'b-o', linewidth=2)
        else:
            plt.plot(time_s, k_mag, 'b-o', linewidth=2)
            
        plt.title("Absolute Permeability Reduction")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Permeability (Darcy)")
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig("outputs/analytics/permeability_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Velocity Distribution Shift 
    plt.figure(figsize=(8, 6))
    v0_valid = (vel_t0[vel_t0 > 1e-6] * u_scale_mms) if vel_t0 is not None else []
    vf_valid = (vel_tfinal[vel_tfinal > 1e-6] * u_scale_mms) if vel_tfinal is not None else []
    
    if len(v0_valid) > 0:
        plt.hist(v0_valid, bins=50, alpha=0.5, label='Initial', density=True, color='blue')
    if len(vf_valid) > 0:
        plt.hist(vf_valid, bins=50, alpha=0.5, label='Clogged', density=True, color='red')
        
    plt.title("Pore Velocity Distribution Shift")
    plt.xlabel("Local Velocity Magnitude (mm/s)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/analytics/velocity_distribution_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Morphology Trajectory 
    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        surface_area = objects[:, 4]
        vol_mm3 = kinetics[:, 2] 
        
        sa_v_ratio = surface_area / (vol_mm3 + 1e-8)
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'tab:red'
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color1)
        ax1.plot(time_s, vol_mm3, color=color1, linewidth=2, marker='s', label='Volume')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'tab:blue'
        ax2.set_ylabel('Surface Area / Volume Ratio (SA/V)', color=color2)  
        ax2.plot(time_s, sa_v_ratio, color=color2, linewidth=2, marker='o', label='SA/V Ratio')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Morphology Trajectory: Patchy vs Layer-like Growth")
        fig.tight_layout()  
        plt.savefig("outputs/analytics/morphology_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Transport Regime (Pe vs Da) 
    if pe_da_data is not None:
        Pe_vals, Da_vals = pe_da_data
        valid_mask = (Pe_vals > 0) & (Da_vals > 0)
        Pe_valid = Pe_vals[valid_mask]
        Da_valid = Da_vals[valid_mask]
        
        plt.figure(figsize=(8, 6))
        num_points = min(5000, len(Pe_valid))
        if num_points > 0:
            idx = np.random.choice(len(Pe_valid), num_points, replace=False)
            plt.scatter(Da_valid[idx], Pe_valid[idx], alpha=0.4, c='purple', s=15, edgecolors='none')
            
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Local Transport Regime ($Pe$ vs $Da$)")
        plt.xlabel("Damköhler Number ($Da$) - Reaction Dominance")
        plt.ylabel("Péclet Number ($Pe$) - Advection Dominance")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.8)
        plt.axvline(x=1, color='k', linestyle='-', alpha=0.8)
        plt.text(0.01, 10, 'Advection-Limited', fontsize=10, color='darkgreen')
        plt.text(10, 0.01, 'Reaction-Limited', fontsize=10, color='darkred')
        plt.savefig("outputs/analytics/transport_regime_da_pe.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # 5. พล็อตกราฟใหม่ล่าสุด! Nucleation Saturation (Dual Y-Axis)
    # ---------------------------------------------------------
    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        num_crystals = objects[:, 1] # คอลัมน์ Number_of_Crystals
        vol_mm3 = kinetics[:, 2] 
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # แกนซ้าย: Number of Crystals (สีเขียว)
        color1 = 'tab:green'
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Number of Crystals (Nucleation Sites)', color=color1)
        ax1.plot(time_s, num_crystals, color=color1, linewidth=2, marker='^', label='Crystal Count')
        ax1.tick_params(axis='y', labelcolor=color1)

        # แกนขวา: Total Volume (สีแดง)
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color2)  
        ax2.plot(time_s, vol_mm3, color=color2, linewidth=2, marker='s', label='Volume')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Nucleation Saturation: Crystal Count & Volume vs Time")
        fig.tight_layout()  
        plt.savefig("outputs/analytics/nucleation_saturation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All analytical PNGs (Physical Units) exported successfully!")

if __name__ == "__main__":
    vel_t0, vel_tfinal, pe_da_data, maps_data, mask_np, u_scale_mms = run_simulation()
    generate_reaction_maps(maps_data, mask_np)
    generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms)