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
import datetime
import scipy.ndimage as ndimage
import pyvista as pv

sys.path.append(os.path.abspath("../"))

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
from src.physics.viscosity import CuSO4ViscositySimulator

def parse_ui_args():
    parser = argparse.ArgumentParser(description="JAX-LaB CuSO4 (v5 PR-EOS MCMP)")
    parser.add_argument("--geom", type=str, default="geometry_mask.npy")
    parser.add_argument("--axis", type=str, choices=['X', 'Y', 'Z'], default='X')
    parser.add_argument("--flow_rate", type=float, default=1.0, help="Flow rate in mL/hr")
    parser.add_argument("--dx_um", type=float, default=20.0, help="Voxel size in micrometers (um)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--inject_size", type=int, default=60)
    parser.add_argument("--t_phys", type=float, default=None,
                        help="Total physical simulation time in seconds. "
                             "If omitted, derived from --steps using a reference tau_f=1.0 dt.")
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

# NOTE: calculate_tau_f (table-interpolation) has been replaced by the
# Price–Davenport empirical model via CuSO4ViscositySimulator.
# The JIT-compiled `update_dynamic_relaxation_time` closure (defined inside
# run_simulation after acoustic scaling is complete) provides the full
# spatially-varying τ_f field.


# =========================================================================
# Geometry-Aware Circular Port Acoustic Scaler
# Guarantees u_lb ≤ target_ulb_max by deriving dt and n_steps from the
# true bottleneck interstitial velocity at the injection face, then
# recalibrates τ_f, τ_c, τ_t to preserve macroscopic ν, D, α exactly.
# =========================================================================
class CircularAcousticScaler:
    """Derives a Mach-safe dt from the pore-scale bottleneck velocity.

    Parameters
    ----------
    dx : float
        Voxel size in metres.
    target_ulb_max : float
        Maximum allowed lattice velocity (default 0.02 for low-Mach stability).
    cs2 : float
        Lattice speed-of-sound squared (1/3 for standard D3Q19).
    """
    def __init__(self, dx: float, target_ulb_max: float = 0.02, cs2: float = 1.0 / 3.0):
        self.dx = float(dx)
        self.target_ulb_max = float(target_ulb_max)
        self.cs2 = float(cs2)

    def recalibrate(
        self,
        Q_ml_hr: float,
        geometry_array: np.ndarray,
        inject_radius_m: float,
        nu_phys: float,
        D_phys: float,
        alpha_phys: float,
        t_phys_total: float,
    ):
        """Compute Mach-safe dt, total LBM steps, and transport-invariant τ values.

        Parameters
        ----------
        Q_ml_hr        : volumetric flow rate in mL/hr
        geometry_array : boolean 3-D array, True = fluid
        inject_radius_m: physical radius of the injection port in metres
        nu_phys        : kinematic viscosity of the fluid (m^2/s)
        D_phys         : solute mass diffusivity (m^2/s)
        alpha_phys     : thermal diffusivity (m^2/s)
        t_phys_total   : desired simulation duration in seconds
        """
        # 1. Convert flow rate to m^3/s
        Q_m3_s = Q_ml_hr * (1e-6 / 3600.0)

        # 2. Build circular mask for the inlet face (first x-plane)
        inlet_slice = geometry_array[0, :, :]      # shape (Ny, Nz)
        Ny, Nz = inlet_slice.shape
        cy_c, cz_c = Ny // 2, Nz // 2
        y_idx, z_idx = np.ogrid[:Ny, :Nz]
        distance_m = np.sqrt((y_idx - cy_c) ** 2 + (z_idx - cz_c) ** 2) * self.dx
        circular_mask = distance_m <= inject_radius_m

        # 3. Effective open area: fluid nodes only inside the circular port
        fluid_nodes_in_circle = inlet_slice & circular_mask
        N_eff = int(np.sum(fluid_nodes_in_circle))
        if N_eff == 0:
            raise ValueError(
                "CircularAcousticScaler: zero fluid nodes in the injection port. "
                "Increase inject_size or check the geometry mask."
            )
        A_eff = N_eff * (self.dx ** 2)          # m^2

        # 4. Maximum bottleneck interstitial velocity through fluid pores only
        V_pore_max = Q_m3_s / A_eff             # m/s

        # 5. Acoustic (diffusive) scaling: choose dt so u_lb = target_ulb_max exactly
        dt = (self.target_ulb_max * self.dx) / V_pore_max
        n_steps = int(np.ceil(t_phys_total / dt))

        # 6. Transport-invariant relaxation recalibration
        #    τ = 0.5 + (transport_coeff * dt) / (cs2 * dx^2)
        tau_f = 0.5 + (nu_phys * dt)    / (self.cs2 * self.dx ** 2)
        tau_c = 0.5 + (D_phys  * dt)    / (self.cs2 * self.dx ** 2)
        tau_t = 0.5 + (alpha_phys * dt) / (self.cs2 * self.dx ** 2)

        print("\n=== Circular Port Acoustic Scaling (Geometry-Aware) ===")
        print(f"  Inject radius    : {inject_radius_m * 1e3:.3f} mm")
        print(f"  Effective area   : {A_eff:.4e} m^2  ({N_eff} fluid nodes)")
        print(f"  V_pore_max       : {V_pore_max:.4e} m/s  ({V_pore_max * 1e3:.3f} mm/s)")
        print(f"  Acoustic dt      : {dt:.4e} s  (u_lb = {self.target_ulb_max})")
        print(f"  Total LBM steps  : {n_steps}  (covers {t_phys_total:.4e} s physical)")
        print(f"  Recalibrated τ   : f={tau_f:.4f}  c={tau_c:.4f}  t={tau_t:.4f}")
        print("======================================================\n")

        return dt, n_steps, tau_f, tau_c, tau_t, V_pore_max


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
        # ป้องกันไม่ให้ความหนาแน่นต่ำกว่า 1e-4 เด็ดขาด 
        # เพื่อให้มีมวลหล่อลื่นเหลืออยู่เสมอ ป้องกันการหารศูนย์
        rho_tree = jax_map(lambda rho: jnp.maximum(rho, 1e-4), rho_tree)
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
            oy, oz = np.where(self.pore_mask_np[-1])
            n_out = len(oy)
            if n_out > 0:
                outlet_idx = (np.full(n_out, self.nx - 1, dtype=int), oy, oz)
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
    radius_vox = args.inject_size / 2.0          # injection port radius in voxels
    cy, cz = ny / 2.0, nz / 2.0                 # centre of the YZ inlet face
    r_sq = (Y - cy)**2 + (Z - cz)**2
    circular_mask_np = r_sq <= radius_vox**2
    circular_mask = jnp.array(circular_mask_np) 
    
    lattice = LatticeD3Q19("f64/f64")
    c_int = np.array(lattice.c, dtype=int).T.tolist()   
    c = jnp.array(lattice.c, dtype=jnp.float64).T       
    w = jnp.array(lattice.w, dtype=jnp.float64)
    c_np = np.array(lattice.c, dtype=np.float64).T 
    
    dx_m = args.dx_um * 1e-6    
    dx_mm = args.dx_um * 1e-3   

    # ---- Physical transport coefficients ----
    nu_phys    = 1e-6       # kinematic viscosity of water at 25°C  [m^2/s]
    D_phys     = 1.0e-9     # CuSO4 solute diffusivity in water     [m^2/s]
    alpha_phys = 1.4e-7     # thermal diffusivity of water           [m^2/s]

    k_r = 0.15
    Q_m3s = args.flow_rate / 3.6e9                              # [m^3/s]
    inject_radius_m = radius_vox * dx_m                         # [m]

    # Physical simulation duration: use --t_phys if given, otherwise derive
    # from --steps using the reference dt (tau_f=1.0 at the current nu_phys).
    _nu_lb_ref = (1.0 - 0.5) / 3.0
    _dt_ref    = (_nu_lb_ref * dx_m ** 2) / nu_phys
    t_phys_total = args.t_phys if args.t_phys is not None else args.steps * _dt_ref

    # ---- Geometry-Aware Circular Port Acoustic Scaling ----
    mask_np_cropped = np.array(mask, dtype=bool)
    scaler = CircularAcousticScaler(dx=dx_m, target_ulb_max=0.02)
    dt_s, n_steps, tau_f_ref, tau_c, tau_t = scaler.recalibrate(
        Q_ml_hr      = args.flow_rate,
        geometry_array = mask_np_cropped,
        inject_radius_m = inject_radius_m,
        nu_phys      = nu_phys,
        D_phys       = D_phys,
        alpha_phys   = alpha_phys,
        t_phys_total = t_phys_total,
    )[:5]   # discard V_pore_max diagnostic return value

    # Relaxation frequencies derived from recalibrated τ values
    omega_t, omega_c = 1.0 / tau_t, 1.0 / tau_c

    # ---------------------------------------------------------------
    # Price–Davenport viscosity: unit-conversion and τ_f closure
    # C_field stores g CuSO4 per 100 mL of water (g/100mL).
    # The PD model requires elemental Cu in g/L → multiply by this factor.
    # ---------------------------------------------------------------
    _CUSO4_TO_CU_GL = 10.0 * (CuSO4ViscositySimulator.MW_CU / CuSO4ViscositySimulator.MW_CUSO4)
    _cs2_val        = 1.0 / 3.0   # D3Q19 lattice speed-of-sound squared

    @jit
    def update_dynamic_relaxation_time(T_field_c, C_field_gL_Cu):
        """
        Spatially-varying τ_f field derived from the Price–Davenport model.

        Parameters
        ----------
        T_field_c      : JAX array of local temperature [°C]
        C_field_gL_Cu  : JAX array of elemental Cu concentration [g/L]

        Returns
        -------
        tau_f_field : JAX array, same spatial shape.  Clamped to (0.5, 10]
                      to guarantee LBM stability everywhere in the domain.
        """
        nu_cSt   = CuSO4ViscositySimulator.calculate_kinematic_viscosity_cSt(
                       T_field_c, C_field_gL_Cu)
        nu_m2s   = nu_cSt * 1e-6                           # cSt → m²/s
        nu_LB    = nu_m2s * (dt_s / (dx_m ** 2))           # lattice units
        tau_f_fld = (nu_LB / _cs2_val) + 0.5
        # τ > 0.5 strictly required; cap at 10 to avoid stagnation artifacts
        return jnp.clip(tau_f_fld, 0.5001, 10.0)

    # Lattice inlet velocity is exactly target_ulb_max = 0.02 by construction
    u_lb = scaler.target_ulb_max
    nu_lb = (tau_f_ref - 0.5) / 3.0

    vol_scale_mm3 = dx_mm ** 3
    k_scale_m2    = dx_m ** 2
    k_scale_darcy = k_scale_m2 / 0.9869233e-12
    u_scale_mms   = dx_mm / dt_s

    # Back-calculate physical inlet velocity for diagnostics
    u_phys_inlet = u_lb * dx_m / dt_s

    print("--- Physical Scales Confirmed (MCMP PR-EOS Mode) ---")
    print(f"  Voxel Size      : {args.dx_um} um")
    print(f"  Acoustic dt     : {dt_s:.4e} s")
    print(f"  Total LBM steps : {n_steps}  (physical time = {t_phys_total:.4e} s)")
    print(f"  Inlet Velocity  : {u_phys_inlet*1000:.4f} mm/s  (u_lb = {u_lb:.4f})")
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
        [0.0, 0.15], 
        [0.15, 0.0]
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
    
    # Phase 1: Residual Solution 
    rho1 = jnp.where(mask, 0.05, 0.0).astype(jnp.float64) 
    # Phase 2: Bulk Air
    rho2 = jnp.where(mask, 0.50, 0.0).astype(jnp.float64) 
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
        u_eq = jnp.nan_to_num(u_eq, nan=0.0, posinf=0.1, neginf=0.1)
        rho1, rho2 = rho_tree

        C_curr = jnp.nan_to_num(jnp.sum(h, axis=-1), nan=0.0, posinf=2.0, neginf=0.0)

        # 2. [COLLISION] MCMP MRT (JAX-LaB) + BGK concentration
        f_post_tree = sim.collision(f_tree)
        f1_post, f2_post = f_post_tree

        # Concentration BGK collision with Stokes-Einstein corrected diffusivity.
        # The local kinematic viscosity (Price-Davenport) modulates D via:
        #   D_local = D_phys * (nu_phys_ref / nu_phys_local)   [Stokes-Einstein]
        # This yields a spatially-varying omega_c_field across the domain.
        C_curr_gL_Cu  = C_curr * _CUSO4_TO_CU_GL
        tau_f_field   = update_dynamic_relaxation_time(T_curr, C_curr_gL_Cu)
        # Invert tau_f_field to recover local nu [m²/s] for SE correction
        nu_local_m2s  = (tau_f_field - 0.5) * _cs2_val * (dx_m ** 2 / dt_s)
        se_ratio      = nu_phys / jnp.maximum(nu_local_m2s, 1e-12)  # nu_ref / nu_local
        D_local       = D_phys * se_ratio
        tau_c_field   = jnp.clip(0.5 + D_local * (dt_s / (_cs2_val * dx_m ** 2)), 0.5001, 10.0)
        omega_c_field = (1.0 / tau_c_field)[..., None]   # broadcast over velocity dim
        h_post = h - omega_c_field * (h - calc_equilibrium_single(C_curr, u_eq))

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
        inlet_open_pores = circular_mask & mask[0]
        if args.axis == 'X':
            u_in     = jnp.zeros_like(u_eq[0]).at[..., 0].set(current_u_lb)
            # Inject Bulk Solution (Component 1) and Residual Air (Component 2)
            f1_eq_in = calc_equilibrium_single(jnp.full_like(rho1[0, ..., 0], 0.50), u_in) 
            f2_eq_in = calc_equilibrium_single(jnp.full_like(rho2[0, ..., 0], 0.05), u_in) 
            h_eq_in  = calc_equilibrium_single(jnp.ones_like(C_curr[0]) * C_inlet, u_in)

            f1_str = f1_str.at[0].set(jnp.where(inlet_open_pores[..., None], f1_eq_in, f1_str[0]))
            f2_str = f2_str.at[0].set(jnp.where(inlet_open_pores[..., None], f2_eq_in, f2_str[0]))
            h_str  = h_str.at[0].set( jnp.where(inlet_open_pores[..., None], h_eq_in,  h_str[0]))

            outlet_open_pores = mask[-1]
            f1_str = f1_str.at[-1].set(jnp.where(outlet_open_pores[..., None], f1_str[-2], f1_str[-1]))
            f2_str = f2_str.at[-1].set(jnp.where(outlet_open_pores[..., None], f2_str[-2], f2_str[-1]))
            h_str  = h_str.at[-1].set( jnp.where(outlet_open_pores[..., None], h_str[-2],  h_str[-1]))

        f1_str    = jnp.nan_to_num(f1_str,   nan=0.0, posinf=1e6,  neginf=-1e6)
        f2_str    = jnp.nan_to_num(f2_str,   nan=0.0, posinf=1e6,  neginf=-1e6)
        h_str     = jnp.nan_to_num(h_str,    nan=0.0, posinf=1e6,  neginf=-1e6)
        solid_frac = jnp.clip(jnp.nan_to_num(solid_frac, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        f_tree_out = [f1_str, f2_str]
        # u_eq is returned so the outer loop can pass it to thermal_solver.step()
        return f_tree_out, h_str, solid_frac, u_eq

    # สร้าง Timestamp Folder เช่น "outputs/run_20261025_143000"
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"outputs/run_{timestamp_str}"

    os.makedirs(f"{out_dir}/vti", exist_ok=True)
    os.makedirs(f"{out_dir}/analytics", exist_ok=True)
   
    # ----------------------------------------------------
    # [NEW] สร้างไฟล์ run_summary.txt เพื่อเก็บ Parameters
    # ----------------------------------------------------
    summary_path = os.path.join(out_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        f_sum.write("=================================================\n")
        f_sum.write("      JAX-LaB CuSO4 Simulation Run Summary       \n")
        f_sum.write("=================================================\n")
        f_sum.write(f"Run Timestamp : {timestamp_str}\n")
        f_sum.write(f"Output Folder : {out_dir}\n\n")
        
        f_sum.write("--- 1. User Input Arguments ---\n")
        f_sum.write(f"Geometry File      : {args.geom}\n")
        f_sum.write(f"Flow Axis          : {args.axis}\n")
        f_sum.write(f"Flow Rate          : {args.flow_rate} mL/hr\n")
        f_sum.write(f"Voxel Size (dx)    : {args.dx_um} um\n")
        f_sum.write(f"Total LBM Steps    : {n_steps}  (physical {t_phys_total:.4e} s)\n")
        f_sum.write(f"Injection Size     : {args.inject_size}^3\n\n")
        
        f_sum.write("--- 2. Physical & LBM Scales (Acoustic Scaling) ---\n")
        f_sum.write(f"Domain Grid Size   : {mask.shape[0]} x {mask.shape[1]} x {mask.shape[2]}\n")
        f_sum.write(f"Inject Radius      : {inject_radius_m * 1e3:.3f} mm\n")
        f_sum.write(f"Acoustic dt        : {dt_s:.6e} s\n")
        f_sum.write(f"Physical Duration  : {t_phys_total:.4e} s\n")
        f_sum.write(f"Physical Inlet Vel : {u_phys_inlet * 1000:.4f} mm/s\n")
        f_sum.write(f"LBM Inlet Vel (u)  : {u_lb:.6f}  (bounded = 0.02)\n")
        f_sum.write(f"Kinematic Visc(nu) : {nu_phys:.2e} m^2/s\n")
        f_sum.write(f"Solute Diff (D)    : {D_phys:.2e} m^2/s\n")
        f_sum.write(f"Thermal Diff (a)   : {alpha_phys:.2e} m^2/s\n\n")

        f_sum.write("--- 3. Thermodynamics & Kinetics ---\n")
        f_sum.write(f"Inlet Temp (T_hot) : {75.0} C\n")
        f_sum.write(f"Wall Temp (T_cold) : {25.0} C\n")
        f_sum.write(f"CHT eta            : 0.15\n")
        f_sum.write(f"Inlet Conc (C_in)  : 60.0 g/100mL\n")
        f_sum.write(f"Reaction Rate (k_r): {k_r}\n\n")

        f_sum.write("--- 4. Transport-Invariant Relaxation Parameters ---\n")
        f_sum.write(f"Fluid  tau_f       : {tau_f_ref:.6f}  (nu_lb = {nu_lb:.6f})\n")
        f_sum.write(f"Thermal tau_t      : {tau_t:.6f}\n")
        f_sum.write(f"Solute  tau_c      : {tau_c:.6f}\n")
        f_sum.write("=================================================\n")
        
    print(f"\n[INFO] Simulation parameters saved to: {summary_path}\n")

    with open(f"{out_dir}/global_kinetics.csv", "w", newline="") as f_csv1, \
         open(f"{out_dir}/object_analysis.csv", "w", newline="") as f_csv2, \
         open(f"{out_dir}/pore_clogging_stats.csv", "w", newline="") as f_csv3:
        csv.writer(f_csv1).writerow(["Step", "Time_s", "Total_Solid_Volume_mm3", "Porosity", "Global_Permeability_Darcy", "Avg_Temperature"])
        csv.writer(f_csv2).writerow(["Step", "Number_of_Crystals", "Avg_Crystal_Size", "Max_Crystal_Size", "Surface_Area"])
        csv.writer(f_csv3).writerow(["Step", "Min_Throat_Size", "Tortuosity_Index"])

    domain_length = float(args.inject_size)
    D_solute = (1.0/3.0) * (tau_c - 0.5)

    print(f"Running Reactive MCMP (PR-EOS) + Thermal-class CHT, {n_steps} LBM steps...")
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

            save_vti_file(f"{out_dir}/vti/precipitate_growth_t{current_step}.vti", binary_precipitate, "CuSO4_Solid")
            save_vti_file(f"{out_dir}/vti/velocity_evolution_t{current_step}.vti", u_np, "Velocity", is_vector=True)
            save_vti_file(f"{out_dir}/vti/supersaturation_map_t{current_step}.vti", supersat_map, "Supersaturation")
            save_vti_file(f"{out_dir}/vti/cuso4_phase_t{current_step}.vti", rho1_np / safe_rho_tot, "CuSO4_Phase")

            P_in = np.mean(rho_tot_np[0][circular_mask_np]) / 3.0
            P_out = np.mean(rho_tot_np[-1][circular_mask_np]) / 3.0
            delta_P = P_in - P_out
            mean_u = np.mean(u_np[..., 0]) 
            
            avg_T      = np.mean(T_np)
            avg_C_gL_Cu = float(np.mean(C_np[fluid_mask_current])) * _CUSO4_TO_CU_GL if np.any(fluid_mask_current) else 0.0
            # Price–Davenport: volume-averaged τ_f from local T and Cu concentration
            tau_f_avg   = float(jnp.mean(
                update_dynamic_relaxation_time(jnp.array(avg_T), jnp.array(avg_C_gL_Cu))
            ))
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

            with open(f"{out_dir}/global_kinetics.csv", "a", newline="") as f_csv1, \
                 open(f"{out_dir}/object_analysis.csv", "a", newline="") as f_csv2, \
                 open(f"{out_dir}/pore_clogging_stats.csv", "a", newline="") as f_csv3:
                csv.writer(f_csv1).writerow([current_step, time_s, current_solid_vol_mm3, porosity, k_perm_darcy, avg_T])
                csv.writer(f_csv2).writerow([current_step, num_features, avg_size, max_size, surface_area])
                csv.writer(f_csv3).writerow([current_step, min_throat, tortuosity])

            print(f"Step {current_step}/{n_steps} | Time: {time_s:.2f} s | Porosity: {porosity:.4f} | Crystals: {num_features} | k: {k_perm_darcy:.2e} Darcy | Max Supersat: {max_supersat:.4f} | Crystal Vol: {current_solid_vol_mm3:.2e} mm3")

        if current_step >= n_steps:
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

    return (vel_mag_t0, vel_mag_tfinal, pe_da_data, maps_data, mask_cpu, u_scale_mms, out_dir)

def generate_reaction_maps(maps_data, mask_np, out_dir):
    print(f"\nGenerating Spatial Reaction Maps in {out_dir}/analytics ...")
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
            
            # Masking non-fluid nodes for visualization
            T_slice[~mask_slice] = np.nan
            C_slice[~mask_slice] = np.nan
            solid_slice[~mask_slice] = np.nan
            
            # 1. Temperature Map (25°C to 75°C)
            im0 = axes[row_idx, 0].imshow(T_slice.T, cmap='inferno', origin='lower', vmin=25, vmax=75)
            axes[row_idx, 0].set_title(f'Step {step}: Temp (°C)')
            fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
            
            # 2. Concentration Map (0 to 65 g/100mL)
            im1 = axes[row_idx, 1].imshow(C_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=65.0)
            axes[row_idx, 1].set_title(f'Step {step}: CuSO\u2084 Conc. (g/100mL)')
            fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
            
            # 3. Solid Fraction Map
            im2 = axes[row_idx, 2].imshow(solid_slice.T, cmap='YlGnBu', origin='lower', vmin=0, vmax=1.0)
            axes[row_idx, 2].set_title(f'Step {step}: Crystal Vol Fraction')
            fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
            
        fig.suptitle(f"Reactive Transport Evolution: {plane_info['title']}", fontsize=18, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f"{out_dir}/analytics/cuso4_reaction_maps_{plane_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Z-Projection Map
    fig_proj, axes_proj = plt.subplots(1, len(steps_to_plot), figsize=(6 * len(steps_to_plot), 5))
    if len(steps_to_plot) == 1: axes_proj = [axes_proj]

    pore_depth = np.sum(mask_np, axis=2).astype(float)
    pore_depth[pore_depth == 0] = np.nan 

    for col_idx, step in enumerate(steps_to_plot):
        solid_sum = maps_data[step]['Z_proj']['solid_sum'].astype(float)
        solid_sum[np.isnan(pore_depth)] = np.nan 

        im = axes_proj[col_idx].imshow(solid_sum.T, cmap='cividis', origin='lower')
        axes_proj[col_idx].set_title(f'Step {step}: Total Crystal Depth')
        fig_proj.colorbar(im, ax=axes_proj[col_idx], fraction=0.046, pad=0.04)

    fig_proj.suptitle("Z-Projection (Top-down Sum of Crystal Volume)", fontsize=18, fontweight='bold')
    fig_proj.tight_layout()
    fig_proj.savefig(f"{out_dir}/analytics/cuso4_reaction_maps_Z_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved Z-Projection Map")

def generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms, out_dir):
    print(f"Generating Analytical PNG Plots in {out_dir}/analytics ...")
    
    # Load CSVs from the specific timestamped directory
    kinetics = np.genfromtxt(f"{out_dir}/global_kinetics.csv", delimiter=',', skip_header=1)
    objects = np.genfromtxt(f"{out_dir}/object_analysis.csv", delimiter=',', skip_header=1)
    
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
            
        plt.title("Absolute Permeability Reduction", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Seconds)", fontsize=12)
        plt.ylabel("Permeability (Darcy)", fontsize=12)
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(f"{out_dir}/analytics/permeability_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(8, 6))
    v0_valid = (vel_t0[vel_t0 > 1e-6] * u_scale_mms) if vel_t0 is not None else []
    vf_valid = (vel_tfinal[vel_tfinal > 1e-6] * u_scale_mms) if vel_tfinal is not None else []

    # [FIX] Calculate a shared global maximum to ensure bin widths match perfectly
    max_v = 0.0
    if len(v0_valid) > 0: max_v = max(max_v, np.max(v0_valid))
    if len(vf_valid) > 0: max_v = max(max_v, np.max(vf_valid))

    # [FIX] Force both histograms to use the exact same 50 bins
    shared_bins = np.linspace(0, max_v, 50)

    if len(v0_valid) > 0:
        plt.hist(v0_valid, bins=shared_bins, alpha=0.6, label='Initial (Pre-Clogging)', density=True, color='royalblue')
    if len(vf_valid) > 0:
        plt.hist(vf_valid, bins=shared_bins, alpha=0.6, label='Final (Clogged)', density=True, color='crimson')

    plt.title("Pore Velocity Distribution Shift", fontsize=14, fontweight='bold')
    plt.xlabel("Local Velocity Magnitude (mm/s)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, ls=":", alpha=0.7)
    plt.savefig(f"{out_dir}/analytics/velocity_distribution_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        surface_area = objects[:, 4]
        vol_mm3 = kinetics[:, 2] 
        
        sa_v_ratio = surface_area / (vol_mm3 + 1e-8)
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'crimson'
        ax1.set_xlabel('Time (Seconds)', fontsize=12)
        ax1.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color1, fontsize=12)
        ax1.plot(time_s, vol_mm3, color=color1, linewidth=2, marker='s', label='Volume')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'teal'
        ax2.set_ylabel('Surface Area / Volume Ratio (SA/V)', color=color2, fontsize=12)  
        ax2.plot(time_s, sa_v_ratio, color=color2, linewidth=2, marker='o', label='SA/V Ratio')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Morphology Trajectory: Patchy vs Layer-like Growth", fontsize=14, fontweight='bold')
        fig.tight_layout()  
        plt.savefig(f"{out_dir}/analytics/morphology_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()

    if pe_da_data is not None:
        Pe_vals, Da_vals = pe_da_data

        # [FIX] Filter out extreme unphysical outliers (dead/clogged zones with u≈0)
        valid_mask = (Pe_vals > 1e-6) & (Da_vals < 1e6) & (Da_vals > 1e-6)
        Pe_valid = Pe_vals[valid_mask]
        Da_valid = Da_vals[valid_mask]

        plt.figure(figsize=(8, 6))
        num_points = min(5000, len(Pe_valid))
        if num_points > 0:
            idx = np.random.choice(len(Pe_valid), num_points, replace=False)
            plt.scatter(Da_valid[idx], Pe_valid[idx], alpha=0.5, c='darkmagenta', s=15, edgecolors='none')

        plt.xscale('log')
        plt.yscale('log')

        # [FIX] Clamp axes so infinity/zero outliers don't squash the data into a dot
        plt.xlim(left=1e-4, right=1e4)
        plt.ylim(bottom=1e-4, top=1e4)

        plt.title("Local Transport Regime ($Pe$ vs $Da$)", fontsize=14, fontweight='bold')
        plt.xlabel("Damk\u00f6hler Number ($Da$) - Reaction Dominance", fontsize=12)
        plt.ylabel("P\u00e9clet Number ($Pe$) - Advection Dominance", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.axhline(y=1, color='k', linestyle='-', alpha=0.8)
        plt.axvline(x=1, color='k', linestyle='-', alpha=0.8)
        plt.text(1e-3, 1e1, 'Advection-Limited', fontsize=11, color='darkgreen', fontweight='bold')
        plt.text(1e1, 1e-3, 'Reaction-Limited', fontsize=11, color='darkred', fontweight='bold')
        plt.savefig(f"{out_dir}/analytics/transport_regime_da_pe.png", dpi=300, bbox_inches='tight')
        plt.close()

    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        num_crystals = objects[:, 1] 
        vol_mm3 = kinetics[:, 2] 
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'forestgreen'
        ax1.set_xlabel('Time (Seconds)', fontsize=12)
        ax1.set_ylabel('Number of Crystals (Nucleation Sites)', color=color1, fontsize=12)
        ax1.plot(time_s, num_crystals, color=color1, linewidth=2, marker='^', label='Crystal Count')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'crimson'
        ax2.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color2, fontsize=12)  
        ax2.plot(time_s, vol_mm3, color=color2, linewidth=2, marker='s', label='Volume')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Nucleation Saturation: Crystal Count & Volume vs Time", fontsize=14, fontweight='bold')
        fig.tight_layout()  
        plt.savefig(f"{out_dir}/analytics/nucleation_saturation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All analytical PNGs exported successfully!")

if __name__ == "__main__":
    vel_t0, vel_tfinal, pe_da_data, maps_data, mask_np, u_scale_mms, out_dir = run_simulation()
    generate_reaction_maps(maps_data, mask_np, out_dir)
    generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms, out_dir)