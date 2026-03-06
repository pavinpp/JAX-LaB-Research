import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import jax
# บังคับให้ JAX ทำงานด้วยความละเอียด float64 เพื่อความเสถียรของ MCMP และ MRT
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.tree import map as jax_map
import csv
import os
import sys
import scipy.ndimage as ndimage
import pyvista as pv

sys.path.append(os.path.abspath("../../"))

from src.lattice import LatticeD3Q19
from src.physics.crystallization import compute_heterogeneous_precipitation, calculate_equilibrium_concentration
from src.physics.porous_media import compute_permeability, compute_supersaturation
from src.physics.wettability import compute_virtual_density
from src.physics.stability_utils import stable_pseudopotential_calculation

# -------------------------------------------------------------------
# [JAX-LaB Core Imports]
# -------------------------------------------------------------------
from src.multiphase import MultiphaseMRT
from src.eos import Peng_Robinson
from src.thermal import BGKSim as ThermalBGK
from src.boundary_conditions import BounceBackHalfway, EquilibriumBC, DoNothing

def parse_ui_args():
    parser = argparse.ArgumentParser(description="JAX-LaB CuSO4 (v5 PR-EOS MCMP)")
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
    temp_points = jnp.array([25.0, 35.0, 45.0, 55.0, 65.0, 75.0], dtype=jnp.float64)
    viscosity_points = jnp.array([1.35, 1.08, 0.89, 0.74, 0.63, 0.55], dtype=jnp.float64)
    mu_T = jnp.interp(T_celsius, temp_points, viscosity_points)
    mu_ref = 1.35 
    tau_f = 0.5 + (tau_ref - 0.5) * (mu_T / mu_ref)
    return tau_f

# =========================================================================
# คลาสจำลอง Reactive MCMP Simulator (สืบทอดจาก Multiphase Core)
# =========================================================================
class ReactiveMCMP_Simulator(MultiphaseMRT):
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.solid_mask = ~mask
        self.fluid_mask = mask
        
    def macroscopic_velocity(self, fin_tree, rho_tree):
        # 1. แทรก Wettability: ปรับความหนาแน่นจำลองที่ขอบของแข็ง (มุม 45 องศา)
        rho_tree_wet = jax_map(
            lambda rho: compute_virtual_density(rho, self.solid_mask, self.fluid_mask, theta=jnp.pi/5, phi=1.1, delta_rho=0.05),
            rho_tree
        )

        # 2. ให้ JAX-LaB คำนวณความเร็วต่อ component ด้วย PR-EOS และ Shan-Chen Forces
        rho_tree_safe = jax_map(lambda rho: jnp.maximum(rho, 1e-8), rho_tree_wet)
        u_tree = super().macroscopic_velocity(fin_tree, rho_tree_safe)

        # 3. คำนวณความเร็วรวม (mass-averaged) เพื่อให้เป็น single array
        u_eq = self.compute_total_velocity(rho_tree_safe, u_tree)

        # 4. [STABILITY] จํากัดความเร็ว (Clipping) ป้องกันโค้ดระเบิดช่วงแรก
        u_eq = jnp.clip(u_eq, -0.1, 0.1)
        return u_eq

    def collision(self, fin_tree):
        fin_tree = jax_map(lambda f: self.precisionPolicy.cast_to_compute(f), fin_tree)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        rho_tree = jax_map(lambda rho: jnp.maximum(rho, 1e-8), rho_tree)
        u_tree = jax_map(lambda u: jnp.nan_to_num(jnp.clip(u, -0.2, 0.2), nan=0.0, posinf=0.2, neginf=-0.2), u_tree)

        m_tree = jax_map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        meq_tree = jax_map(lambda feq, M: jnp.dot(feq, M), feq_tree, self.M)

        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.adjust_surface_tension(psi_tree)
        mout_tree = jax_map(
            lambda m, meq, S: m - jnp.dot(m - meq, S),
            m_tree,
            meq_tree,
            self.S,
        )
        mout_tree = self.apply_force(mout_tree, meq_tree, rho_tree, u_tree)
        fout_tree = jax_map(lambda m, Minv, C: jnp.dot(m + C, Minv), mout_tree, self.M_inv, C_tree)
        fout_tree = jax_map(lambda fout: jnp.nan_to_num(fout, nan=0.0, posinf=1e6, neginf=-1e6), fout_tree)
        return jax_map(lambda fout: self.precisionPolicy.cast_to_output(fout), fout_tree)


# =========================================================================
# Robin (Partial Bounce-Back) Thermal BC — Conjugate Heat Transfer
# =========================================================================
class RobinThermalBC(BounceBackHalfway):
    """
    Partial Bounce-Back thermal BC for Conjugate Heat Transfer (CHT).

    Blends adiabatic (full bounce-back) and isothermal (wall-equilibrium) returns:
        g_post = (1 - eta) * g_adiabatic + eta * w_i * T_wall

    eta=0  → purely adiabatic wall (PLA/ABS plastic, k ≈ 0.1–0.25 W/m·K)
    eta=1  → perfectly isothermal wall (glass bead, k ≈ 1.0–1.1 W/m·K)
    """
    def __init__(self, indices, gridInfo, precision_policy, eta=0.15, T_wall=25.0):
        super().__init__(indices, gridInfo, precision_policy)
        self.eta = eta
        self.T_wall = T_wall

    @partial(jit, static_argnums=(0,))
    def apply(self, gout, gin):
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]            # (nbd, 1)
        gbd = gout[self.indices]                    # (nbd, q)
        # Adiabatic: standard half-way bounce-back from pre-streaming opposite direction
        g_adiabatic = gin[self.indices][bindex, self.iknown]   # (nbd, q)
        # Isothermal: equilibrium at T_wall with zero velocity
        w_arr = jnp.array(self.lattice.w, dtype=gout.dtype)
        g_isothermal = w_arr[self.imissing] * self.T_wall      # (nbd, q)
        # Robin blend
        g_robin = (1.0 - self.eta) * g_adiabatic + self.eta * g_isothermal
        gbd = gbd.at[bindex, self.imissing].set(g_robin)
        return gbd


# =========================================================================
# Conjugate Heat Transfer Thermal Solver (BGK, D3Q19)
# =========================================================================
class CuSO4_ThermalSolver(ThermalBGK):
    """
    BGK thermal LBM solver for the CuSO4 imbibition experiment.

    Boundary conditions registered:
      - Solid walls       : RobinThermalBC (CHT partial bounce-back, configurable eta_cht)
      - Inlet (x=0)       : EquilibriumBC at T_hot with u_lb x-velocity
      - Outlet (x=-1)     : DoNothing (zero-gradient outflow)

    The solver is driven externally: `g, _ = thermal_solver.step(g, u_eq, 0)`
    is called once per LBM step from the main Python loop.
    """
    def __init__(self, pore_mask_np, T_hot, T_cold, circular_mask_np,
                 u_lb_val, eta_cht=0.15, **kwargs):
        # Attributes MUST be set before super().__init__() because LBMBase.__init__
        # calls _create_boundary_data() → set_boundary_conditions() at the end of __init__.
        self.pore_mask_np  = np.array(pore_mask_np, dtype=bool)
        self.T_hot         = float(T_hot)
        self.T_cold        = float(T_cold)
        self.circ_mask_np  = np.array(circular_mask_np, dtype=bool)
        self.u_lb_val      = float(u_lb_val)
        self.eta_cht       = float(eta_cht)
        super().__init__(**kwargs)   # triggers _create_boundary_data → set_boundary_conditions

    # ------------------------------------------------------------------
    def set_boundary_conditions(self):
        """Populate self.BCs (for grid-mask) and self.thermal_BCs (for thermal physics)."""
        # self.BCs must be initialised here; Thermal._create_boundary_data never does it.
        self.BCs = []

        # ---- 1. Solid-wall BCs ----
        xs, ys, zs = np.where(~self.pore_mask_np)
        if len(xs) > 0:
            solid_idx = (xs, ys, zs)
            # Fluid placeholder — only used so Thermal._create_boundary_data can build
            # the grid_mask correctly (isSolid=True tells it which voxels are walls).
            self.BCs.append(BounceBackHalfway(
                indices=solid_idx,
                gridInfo=self.gridInfo,
                precision_policy=self.precisionPolicy,
            ))
            # Robin (CHT) thermal BC at all solid-wall nodes
            self.thermal_BCs.append(RobinThermalBC(
                indices=solid_idx,
                gridInfo=self.gridInfo,
                precision_policy=self.precisionPolicy,
                eta=self.eta_cht,
                T_wall=self.T_cold,
            ))

        # ---- 2. Inlet (x=0) — fixed temperature T_hot ----
        iy, iz = np.where(self.circ_mask_np)
        n_in = len(iy)
        if n_in > 0:
            inlet_idx = (np.zeros(n_in, dtype=int), iy, iz)
            T_in  = self.T_hot * jnp.ones((n_in, 1),  dtype=jnp.float64)
            u_in  = jnp.zeros((n_in, 3), dtype=jnp.float64).at[:, 0].set(self.u_lb_val)
            self.thermal_BCs.append(EquilibriumBC(
                indices=inlet_idx,
                gridInfo=self.gridInfo,
                precision_policy=self.precisionPolicy,
                rho=T_in,
                u=u_in,
            ))

            # ---- 3. Outlet (x=-1) — zero-gradient / do-nothing ----
            outlet_idx = (np.full(n_in, self.nx - 1, dtype=int), iy, iz)
            self.thermal_BCs.append(DoNothing(
                indices=outlet_idx,
                gridInfo=self.gridInfo,
                precision_policy=self.precisionPolicy,
            ))


# =========================================================================
# ฟังก์ชันคำนวณ LBM หลัก
# =========================================================================
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
    
    ny, nz = mask.shape[1], mask.shape[2]
    Y, Z = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    radius = 30.0 
    cy, cz = ny / 2.0, radius 
    r_sq = (Y - cy)**2 + (Z - cz)**2
    circular_mask_np = r_sq <= radius**2
    circular_mask = jnp.array(circular_mask_np) 
    
    lattice = LatticeD3Q19("f64/f64")
    c_int = np.array(lattice.c, dtype=int).T.tolist()   
    c = jnp.array(lattice.c, dtype=jnp.float64).T       
    w = jnp.array(lattice.w, dtype=jnp.float64)
    c_np = np.array(lattice.c, dtype=np.float64).T 
    
    dx_m = args.dx_um * 1e-6    
    dx_mm = args.dx_um * 1e-3   
    nu_phys = 1e-6 
    
    tau_f_ref = 1.0   
    tau_t = 0.55  
    tau_c = 0.95  
    k_r = 0.15
    omega_t, omega_c = 1.0/tau_t, 1.0/tau_c
    
    nu_lb = (tau_f_ref - 0.5) / 3.0
    dt_s = (nu_lb * (dx_m ** 2)) / nu_phys 
    
    Q_m3s = args.flow_rate / 3.6e9  
    cross_section_area_m2 = float(np.sum(circular_mask_np)) * (dx_m ** 2)
    u_phys_inlet = Q_m3s / cross_section_area_m2 
    u_lb = u_phys_inlet * (dt_s / dx_m)
    # Cap lattice inlet velocity to 0.02 to stay in the low-Mach stable regime.
    # Flow rates that would exceed this are physically represented at the capped speed;
    # increase --steps proportionally to preserve the same physical duration.
    U_LB_MAX = 0.02
    if u_lb > U_LB_MAX:
        print(f"  [Warning] u_lb={u_lb:.4f} exceeds stability limit {U_LB_MAX}. Capping to {U_LB_MAX}.")
        print(f"  [Hint] Increase --steps by ~{u_lb/U_LB_MAX:.1f}x to maintain physical duration.")
        u_lb = U_LB_MAX

    vol_scale_mm3 = dx_mm ** 3
    k_scale_m2 = dx_m ** 2
    k_scale_darcy = k_scale_m2 / 0.9869233e-12  
    u_scale_mms = (dx_mm / dt_s)                
    
    print("\n--- Physical Scales Confirmed (MCMP PR-EOS Mode) ---")
    print(f"  Voxel Size: {args.dx_um} um")
    print(f"  Time Step (dt): {dt_s:.2e} s")
    print(f"  Inlet Velocity (Target): {u_phys_inlet*1000:.2f} mm/s (LBM: {u_lb:.4f})")
    print("----------------------------------------------------\n")

    # ---------------------------------------------------------
    # ตั้งค่า Native JAX-LaB Core สำหรับ MCMP PR-EOS
    # ---------------------------------------------------------
    # 1. พารามิเตอร์ Peng-Robinson แบบ Safe-mode สำหรับ 2 Components
    # ---------------------------------------------------------
    # ตั้งค่า Native JAX-LaB Core สำหรับ MCMP PR-EOS (Bullet-proof)
    # ---------------------------------------------------------
    
    # 1. พารามิเตอร์ Peng-Robinson (ใส่ให้ครบทุกตัวที่ eos.py ต้องการ)
    pr_eos = Peng_Robinson(
        a=[0.02, 0.02],           # Cohesion parameter 
        b=[0.05, 0.05],           # Co-volume parameter
        R=[1.0, 1.0],             # Gas constant
        Tc=[1.0, 1.0],            # Critical temperature
        pr_omega=[0.344, 0.344],  # Acentric factor (ใช้ชื่อ pr_omega ตามไลบรารี)
        T=0.85                    # Isothermal temperature
    )
    
    # 2. ปฏิสัมพันธ์ Shan-Chen
    g_kkprime_val = jnp.array([
        [0.0, 0.57], 
        [0.57, 0.0]
    ], dtype=jnp.float64)

    # 3. MRT transform matrix for D3Q19
    e = np.array(lattice.c, dtype=np.float64).T
    en = np.linalg.norm(e, axis=1)
    M = np.zeros((19, 19), dtype=np.float64)
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]
    
    # 4. เตรียมขนาดโดเมน (Grid Size) เผื่อ LBMBase ต้องการ
    nx, ny, nz = mask.shape

    # 5. MRT relaxation parameters
    s_rho = [0.0, 0.0]
    s_e = [1.0, 1.0]
    s_eta = [1.0, 1.0]
    s_j = [1.0, 1.0]
    s_q = [1.2, 1.2]
    s_v = [1.0, 1.0]
    s_pi = [1.0, 1.0]
    s_m = [1.0, 1.0]
    
    # 6. สร้าง Object ของ Simulator โดยอัด kwargs ให้ครบทุกระดับ!
    sim = ReactiveMCMP_Simulator(
        # --- Custom Physics Kwargs ---
        mask=mask,

        # --- LBMBase Kwargs (Core) ---
        lattice=lattice,
        precision="f64/f64",
        nx=nx,
        ny=ny,
        nz=nz,

        # --- Multiphase Kwargs ---
        EOS=pr_eos,               # ต้องใช้ชื่อ EOS (ตัวพิมพ์ใหญ่) ตาม Multiphase.__init__
        n_components=2,           # บังคับระบุจำนวน Component อย่างชัดเจน
        k=[1.0, 1.0],             # Modification coefficient สำหรับ EOS potential
        kappa=[0.1, 0.1],         # MRT surface tension tuning
        A=0.1 * np.ones((2, 2)),  # Zhang-Chen weighting: 0.1 improves stability at high density ratios
        g_kkprime=g_kkprime_val,
        s_rho=s_rho,
        s_e=s_e,
        s_eta=s_eta,
        s_j=s_j,
        s_q=s_q,
        s_v=s_v,
        s_pi=s_pi,
        s_m=s_m,
        M=[M, M],
    )

    # ---------------------------------------------------------
    # Initialize State Variables
    # ---------------------------------------------------------
    T_hot, T_cold = 75.0, 25.0
    C_inlet = 60.0  # 60 g/100 mL H2O — undersaturated at 75°C (solubility ~83.8 g/100mL)
    
    rho1 = jnp.full(mask.shape, 1e-4, dtype=jnp.float64)
    rho2 = jnp.ones(mask.shape, dtype=jnp.float64) * 0.5 # Native Air
    u_init = jnp.zeros(mask.shape + (3,), dtype=jnp.float64)
    
    f1 = sim.equilibrium(rho1[..., None], u_init).astype(jnp.float64)
    f2 = sim.equilibrium(rho2[..., None], u_init).astype(jnp.float64)
    f_tree = [f1, f2] # โครงสร้าง PyTree สำหรับ MCMP

    # ---------------------------------------------------------
    # Conjugate Heat Transfer Thermal Solver (Thermal class)
    # ---------------------------------------------------------
    # omega = 1/tau_t for BGK thermal relaxation
    # precipitation kinetics and BCs are wired inside CuSO4_ThermalSolver
    mask_np_for_thermal = np.array(mask, dtype=bool)
    thermal_solver = CuSO4_ThermalSolver(
        pore_mask_np   = mask_np_for_thermal,
        T_hot          = T_hot,
        T_cold         = T_cold,
        circular_mask_np = circular_mask_np,
        u_lb_val       = u_lb,
        eta_cht        = 0.15,
        # --- LBMBase kwargs ---
        lattice        = lattice,
        precision      = "f64/f64",
        nx             = nx,
        ny             = ny,
        nz             = nz,
        omega          = 1.0 / tau_t,
    )
    print("[Thermal] CuSO4_ThermalSolver (BGK, Robin CHT) initialised.")

    T_field = jnp.ones(mask.shape, dtype=jnp.float64) * T_cold
    C_field = jnp.zeros(mask.shape, dtype=jnp.float64)
    solid_frac = jnp.zeros(mask.shape, dtype=jnp.float64)  # alias: replaces solid_fraction

    @jit
    def calc_equilibrium_single(phi, u_eq):
        cu = jnp.dot(u_eq, c.T)
        usqr = jnp.sum(u_eq**2, axis=-1, keepdims=True)
        return phi[..., None] * w * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)

    def shift_no_wrap(a, sx, sy, sz):
        out = jnp.zeros_like(a)
        xs_src = slice(max(-sx, 0), a.shape[0] - max(sx, 0))
        ys_src = slice(max(-sy, 0), a.shape[1] - max(sy, 0))
        zs_src = slice(max(-sz, 0), a.shape[2] - max(sz, 0))
        xs_dst = slice(max(sx, 0), a.shape[0] - max(-sx, 0))
        ys_dst = slice(max(sy, 0), a.shape[1] - max(-sy, 0))
        zs_dst = slice(max(sz, 0), a.shape[2] - max(-sz, 0))
        return out.at[xs_dst, ys_dst, zs_dst].set(a[xs_src, ys_src, zs_src])

    g = calc_equilibrium_single(T_field, u_init)
    h = calc_equilibrium_single(C_field, u_init)
    # T_curr: scalar temperature field driven by thermal_solver each iteration
    T_curr = T_field

    @jit
    def calculate_solubility_curve(T):
        """Equilibrium concentration C_eq(T) in g/100mL H2O.
        2nd-order polynomial fit from:
        (20, 32.0), (40, 44.6), (60, 61.8), (80, 83.8), (100, 114.0)
        """
        a = jnp.float64(0.00642857)
        b = jnp.float64(0.25285714)
        c_coef = jnp.float64(24.34285714)
        return a * (T ** 2) + b * T + c_coef

    @jit
    def lbm_step(f_tree, h, solid_frac, T_curr, step_idx):
        """
        Single MCMP + concentration step.
        Temperature (g) is managed EXTERNALLY by CuSO4_ThermalSolver.
        Returns (f_tree_out, h_out, solid_frac_out, u_eq) where u_eq
        is passed to thermal_solver.step() in the outer Python loop.
        """
        # 1. Update Macroscopic ของไหล
        rho_tree, _ = sim.update_macroscopic(f_tree)
        u_eq = sim.macroscopic_velocity(f_tree, rho_tree)
        u_eq = jnp.nan_to_num(u_eq, nan=0.0, posinf=0.1, neginf=-0.1)
        rho1, rho2 = rho_tree

        C_curr = jnp.nan_to_num(jnp.sum(h, axis=-1), nan=0.0, posinf=2.0, neginf=0.0)

        # 2. [COLLISION] MCMP MRT (JAX-LaB) + BGK concentration
        f_post_tree = sim.collision(f_tree)
        f1_post, f2_post = f_post_tree

        # Concentration BGK collision (T handled by Thermal class)
        h_post = h - omega_c * (h - calc_equilibrium_single(C_curr, u_eq))

        # 3. [PRECIPITATION KINETICS] — polynomial solubility against local T_curr
        effective_fluid_mask = mask & (solid_frac < 0.5)
        wall_mask = jnp.zeros_like(effective_fluid_mask, dtype=bool)
        if args.axis == 'X':
            wall_mask = wall_mask.at[:, 0, :].set(True)
            wall_mask = wall_mask.at[:, -1, :].set(True)
            wall_mask = wall_mask.at[:, :, 0].set(True)
            wall_mask = wall_mask.at[:, :, -1].set(True)
            wall_mask = wall_mask.at[0, :, :].set(wall_mask[0, :, :] | ~circular_mask)
            wall_mask = wall_mask.at[-1, :, :].set(wall_mask[-1, :, :] | ~circular_mask)

        effective_fluid_mask_bc = effective_fluid_mask & (~wall_mask)
        C_eq_local = calculate_solubility_curve(T_curr)
        supersat = jnp.maximum(C_curr - C_eq_local, 0.0)
        delta_C = jnp.where(effective_fluid_mask_bc, k_r * supersat, 0.0)
        h_post = h_post - w * delta_C[..., None]
        solid_frac = solid_frac + delta_C

        # 4. [STREAMING & BOUNCE-BACK] for momentum and concentration only.
        #    Temperature (g) streaming + Robin CHT BC is handled by
        #    CuSO4_ThermalSolver.step() called in the outer Python loop.
        f1_str, f2_str = jnp.zeros_like(f1_post), jnp.zeros_like(f2_post)
        h_str = jnp.zeros_like(h)
        solid_mask_bc = ~effective_fluid_mask_bc
        domain_ones = jnp.ones_like(mask, dtype=bool)
        opp = jnp.array(lattice.opp_indices)

        for i in range(19):
            sx, sy, sz = c_int[i][0], c_int[i][1], c_int[i][2]
            f1_str_i = shift_no_wrap(f1_post[..., i], sx, sy, sz)
            f2_str_i = shift_no_wrap(f2_post[..., i], sx, sy, sz)
            h_str_i  = shift_no_wrap(h_post[..., i],  sx, sy, sz)

            is_invalid = shift_no_wrap(solid_mask_bc, sx, sy, sz) | ~shift_no_wrap(domain_ones, sx, sy, sz)

            f1_str = f1_str.at[..., i].set(jnp.where(is_invalid, f1_post[..., opp[i]], f1_str_i))
            f2_str = f2_str.at[..., i].set(jnp.where(is_invalid, f2_post[..., opp[i]], f2_str_i))
            h_str  = h_str.at[..., i].set(jnp.where(is_invalid, h_post[...,  opp[i]], h_str_i))

        # 5. [INLET/OUTLET BCs] for f and h — soft start ramp
        #    Thermal inlet/outlet are handled by CuSO4_ThermalSolver (EquilibriumBC / DoNothing).
        ramp_factor = jnp.clip(step_idx / 500.0, 0.0, 1.0)
        current_u_lb = u_lb * ramp_factor
        if args.axis == 'X':
            u_in     = jnp.zeros_like(u_eq[0]).at[..., 0].set(current_u_lb)
            f1_eq_in = calc_equilibrium_single(jnp.ones_like(rho1[0, ..., 0]), u_in)
            f2_eq_in = calc_equilibrium_single(jnp.zeros_like(rho2[0, ..., 0]), u_in)
            h_eq_in  = calc_equilibrium_single(jnp.ones_like(C_curr[0]) * C_inlet, u_in)

            f1_str = f1_str.at[0].set(jnp.where(circular_mask[..., None], f1_eq_in, f1_str[0]))
            f2_str = f2_str.at[0].set(jnp.where(circular_mask[..., None], f2_eq_in, f2_str[0]))
            h_str  = h_str.at[0].set( jnp.where(circular_mask[..., None], h_eq_in,  h_str[0]))

            f1_str = f1_str.at[-1].set(jnp.where(circular_mask[..., None], f1_str[-2], f1_str[-1]))
            f2_str = f2_str.at[-1].set(jnp.where(circular_mask[..., None], f2_str[-2], f2_str[-1]))
            h_str  = h_str.at[-1].set( jnp.where(circular_mask[..., None], h_str[-2],  h_str[-1]))

        f1_str    = jnp.nan_to_num(f1_str,   nan=0.0, posinf=1e6,  neginf=-1e6)
        f2_str    = jnp.nan_to_num(f2_str,   nan=0.0, posinf=1e6,  neginf=-1e6)
        h_str     = jnp.nan_to_num(h_str,    nan=0.0, posinf=1e6,  neginf=-1e6)
        solid_frac = jnp.clip(jnp.nan_to_num(solid_frac, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        f_tree_out = [f1_str, f2_str]
        # u_eq is returned so the outer loop can pass it to thermal_solver.step()
        return f_tree_out, h_str, solid_frac, u_eq

    os.makedirs("outputs/vti", exist_ok=True)
    os.makedirs("outputs/analytics", exist_ok=True)
    
    with open("outputs/global_kinetics.csv", "w", newline="") as f_csv1, \
         open("outputs/object_analysis.csv", "w", newline="") as f_csv2, \
         open("outputs/pore_clogging_stats.csv", "w", newline="") as f_csv3:
        csv.writer(f_csv1).writerow(["Step", "Time_s", "Total_Solid_Volume_mm3", "Porosity", "Global_Permeability_Darcy", "Avg_Temperature"])
        csv.writer(f_csv2).writerow(["Step", "Number_of_Crystals", "Avg_Crystal_Size", "Max_Crystal_Size", "Surface_Area"])
        csv.writer(f_csv3).writerow(["Step", "Min_Throat_Size", "Tortuosity_Index"])

    domain_length = float(args.inject_size)
    D_solute = (1.0/3.0) * (tau_c - 0.5)

    print(f"Running Reactive MCMP (PR-EOS) + Thermal-class CHT, {args.steps} LBM steps...")
    # State is now split: (f_tree, h, solid_frac) for MCMP+concentration
    # and `g` for temperature, driven by CuSO4_ThermalSolver each step.
    mask_cpu = np.array(mask)
    mid_x, mid_y, mid_z = mask_cpu.shape[0]//2, mask_cpu.shape[1]//2, mask_cpu.shape[2]//2

    vel_mag_t0    = None
    vel_mag_tfinal = None
    pe_da_data    = []
    maps_data     = {}

    current_step = 0
    while True:
        f_tree[0].block_until_ready()

        if current_step == 0 or current_step % 500 == 0:
            f1_np, f2_np = [np.array(x) for x in f_tree]
            g_np     = np.array(g)
            h_np     = np.array(h)
            solid_np = np.array(solid_frac)
            
            rho1_np = np.sum(f1_np, axis=-1)
            rho2_np = np.sum(f2_np, axis=-1)
            rho_tot_np = rho1_np + rho2_np
            
            safe_rho1 = np.where(rho1_np == 0, 1e-8, rho1_np)
            safe_rho2 = np.where(rho2_np == 0, 1e-8, rho2_np)
            safe_rho_tot = np.where(rho_tot_np == 0, 1e-8, rho_tot_np)
            
            u1_np = np.dot(f1_np, c_np) / safe_rho1[..., None]
            u2_np = np.dot(f2_np, c_np) / safe_rho2[..., None]
            u_np = (rho1_np[..., None] * u1_np + rho2_np[..., None] * u2_np) / safe_rho_tot[..., None]
            
            T_np = np.sum(g_np, axis=-1)
            C_np = np.sum(h_np, axis=-1)
            
            binary_precipitate = np.where(solid_np > 0.1, 1.0, 0.0).astype(np.float32)
            fluid_mask_current = mask_cpu & (solid_np < 0.5)
            supersat_map = np.array(compute_supersaturation(jnp.array(C_np), jnp.array(T_np), calculate_equilibrium_concentration))
            u_mag = np.linalg.norm(u_np, axis=-1)
            
            if current_step == 0:
                vel_mag_t0 = u_mag[fluid_mask_current]

            maps_data[current_step] = {
                'XY': {'T': T_np[:, :, mid_z].copy(), 'C': C_np[:, :, mid_z].copy(), 'solid': solid_np[:, :, mid_z].copy()},
                'XZ': {'T': T_np[:, mid_y, :].copy(), 'C': C_np[:, mid_y, :].copy(), 'solid': solid_np[:, mid_y, :].copy()},
                'YZ': {'T': T_np[mid_x, :, :].copy(), 'C': C_np[mid_x, :, :].copy(), 'solid': solid_np[mid_x, :, :].copy()},
                'Z_proj': {'solid_sum': np.sum(solid_np, axis=2).copy()}
            }

            save_vti_file(f"outputs/vti/precipitate_growth_t{current_step}.vti", binary_precipitate, "CuSO4_Solid")
            save_vti_file(f"outputs/vti/velocity_evolution_t{current_step}.vti", u_np, "Velocity", is_vector=True)
            save_vti_file(f"outputs/vti/supersaturation_map_t{current_step}.vti", supersat_map, "Supersaturation")
            save_vti_file(f"outputs/vti/cuso4_phase_t{current_step}.vti", rho1_np / safe_rho_tot, "CuSO4_Phase")

            P_in = np.mean(rho_tot_np[0][circular_mask_np]) / 3.0
            P_out = np.mean(rho_tot_np[-1][circular_mask_np]) / 3.0
            delta_P = P_in - P_out
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

            # Pseudopotential stability diagnostics (PR-EOS singularity check)
            rho_tree_diag = [jnp.array(rho1_np[..., None]), jnp.array(rho2_np[..., None])]
            from src.physics.stability_utils import log_pseudopotential_stability
            log_pseudopotential_stability(rho_tree_diag, pr_eos, current_step, fluid_mask=mask_cpu)

            with open("outputs/global_kinetics.csv", "a", newline="") as f_csv1, \
                 open("outputs/object_analysis.csv", "a", newline="") as f_csv2, \
                 open("outputs/pore_clogging_stats.csv", "a", newline="") as f_csv3:
                csv.writer(f_csv1).writerow([current_step, time_s, current_solid_vol_mm3, porosity, k_perm_darcy, avg_T])
                csv.writer(f_csv2).writerow([current_step, num_features, avg_size, max_size, surface_area])
                csv.writer(f_csv3).writerow([current_step, min_throat, tortuosity])

            print(f"Step {current_step}/{args.steps} | Time: {time_s:.2f} s | Porosity: {porosity:.4f} | Crystals: {num_features} | k: {k_perm_darcy:.2e} Darcy | Max Supersat: {max_supersat:.4f} | Crystal Vol: {current_solid_vol_mm3:.2e} mm3")

        if current_step >= args.steps:
            break

        # ---- Single LBM step ----
        # a) MCMP + concentration (f1, f2, h) — returns u_eq for thermal
        f_tree, h, solid_frac, u_eq = lbm_step(f_tree, h, solid_frac, T_curr, current_step)
        # b) Temperature (g) — Thermal class: BGK collision + Robin CHT BC + inlet/outlet BCs
        #    timestep=0 is a constant placeholder (none of our BCs are dynamic)
        g, _ = thermal_solver.step(g, u_eq, 0)
        T_curr = jnp.nan_to_num(jnp.sum(g, axis=-1),
                                nan=T_cold, posinf=T_hot, neginf=T_cold)
        current_step += 1

    vel_mag_tfinal = u_mag[fluid_mask_current]
    
    u_mag_safe = np.where(u_mag == 0, 1e-8, u_mag)
    L_ref = 1.0 
    Pe_map = (u_mag * L_ref) / D_solute
    Da_map = (k_r * L_ref) / u_mag_safe
    pe_da_data = (Pe_map[fluid_mask_current], Da_map[fluid_mask_current])

    return (vel_mag_t0, vel_mag_tfinal, pe_da_data, maps_data, mask_cpu, u_scale_mms)

# --- ละโค้ด generate_reaction_maps และ generate_analytical_plots ไว้ด้านล่าง (ใช้โค้ดชุด v3 เดิมได้เลย) ---
def generate_reaction_maps(maps_data, mask_np):
    print("\nGenerating Spatial Reaction Maps (XY, XZ, YZ, and Z-Projection)...")
    steps_saved = sorted(list(maps_data.keys()))
    
    if len(steps_saved) >= 3:
        steps_to_plot = [steps_saved[0], steps_saved[len(steps_saved)//2], steps_saved[-1]]
    else:
        steps_to_plot = steps_saved
        
    mid_x, mid_y, mid_z = mask_np.shape[0]//2, mask_np.shape[1]//2, mask_np.shape[2]//2
    
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

    fig_proj, axes_proj = plt.subplots(1, len(steps_to_plot), figsize=(6 * len(steps_to_plot), 5))
    if len(steps_to_plot) == 1: axes_proj = [axes_proj]

    pore_depth = np.sum(mask_np, axis=2).astype(float)
    pore_depth[pore_depth == 0] = np.nan 

    for col_idx, step in enumerate(steps_to_plot):
        solid_sum = maps_data[step]['Z_proj']['solid_sum'].astype(float)
        solid_sum[np.isnan(pore_depth)] = np.nan 

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

    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        num_crystals = objects[:, 1] 
        vol_mm3 = kinetics[:, 2] 
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'tab:green'
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Number of Crystals (Nucleation Sites)', color=color1)
        ax1.plot(time_s, num_crystals, color=color1, linewidth=2, marker='^', label='Crystal Count')
        ax1.tick_params(axis='y', labelcolor=color1)

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