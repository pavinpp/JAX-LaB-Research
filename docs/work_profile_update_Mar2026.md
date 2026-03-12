# JAX-LaB CuSO₄ Simulation — Development Update
**Date:** March 2026 | **Project:** CuSO₄ Crystallization in Porous Media (LBM / JAX-LaB)

---

## Slide 1 — Physics Model Evolution

| Version | Key Addition | Physics Significance |
|---------|-------------|----------------------|
| **v0** `fix_viscosity` | Temperature-dependent viscosity via interpolation table | τ_f dynamically scales with T (25–75 °C) using CuSO₄ viscosity data |
| **v1** `z_projection` | Selectable flow axis (X / Y / Z) | Geometry-agnostic inlet/outlet; supports arbitrary scan orientations |
| **v2** `circular_inlet` | Circular cross-section inlet mask | Matches physical tube geometry; removes corner artifacts at injection face |
| **v3** `wettability` | Surface wettability via virtual density | Contact angle (θ) enforcement at solid–fluid interface using `compute_virtual_density` |
| **v4** `MRT_only` | Full MRT collision operator (D3Q19) | Orthogonal transformation matrix M constructed in float64; improved stability over single-τ BGK |
| **v5** `EOS` | Peng–Robinson EOS + Multi-Component Multiphase (MCMP) | Two-fluid Shan–Chen interactions with thermodynamically consistent equation of state; JAX float64 enabled |
| **v6** `Thermal` | Coupled thermal transport (BGKSim) | Full thermo-hydrodynamic coupling; BounceBack / Equilibrium / DoNothing boundary conditions |
| **v7** `AcousticScaler` | `CircularAcousticScaler` class | Auto-derives Mach-safe dt from pore-scale bottleneck velocity; recalibrates τ_f, τ_c, τ_t to preserve physical ν, D, α |
| **latest** `simulate_CuSO4.py` | `CuSO4ViscositySimulator` (Price–Davenport model) + `--t_phys` argument | Replaces table interpolation with empirical viscosity model; user can specify total physical time directly |

---

## Slide 2 — Architecture & Current Capabilities

### What the simulation now does end-to-end

```
.npy geometry (micro-CT scan)
        │
        ▼
  CircularAcousticScaler          ← guarantees u_lb ≤ 0.02 (low-Mach stable)
        │  derives dt, n_steps, recalibrates τ_f / τ_c / τ_t
        ▼
  ReactiveMCMP_Simulator          ← extends JAX-LaB MultiphaseMRT
  ├─ PR-EOS (Peng–Robinson)       ← thermodynamic pressure tensor
  ├─ MRT collision (D3Q19)        ← numerical stability
  ├─ Wettability (virtual ρ)      ← contact angle at pore walls
  ├─ CuSO4ViscositySimulator      ← spatially-varying τ_f(x,T)
  └─ Thermal BGK                  ← coupled temperature field
        │
        ▼
  Heterogeneous Precipitation     ← crystal nucleation & growth model
  Permeability tracking           ← Darcy-scale output via compute_permeability
        │
        ▼
  Output: .vti files (ParaView)   ← velocity, concentration, solid fraction, temperature
          .csv log                ← permeability, supersaturation, crystal volume over time
```

### Key improvements over baseline
- **Stability** — MRT + acoustic scaling + velocity clipping prevent divergence in complex pore geometries
- **Physical accuracy** — PR-EOS captures real fluid compressibility; Price–Davenport viscosity matches CuSO₄ experimental data
- **Flexibility** — `--t_phys` maps LBM steps to real seconds automatically; axis-selectable flow direction
- **GPU-ready** — Full JAX JIT compilation; float64 precision for multiphase stability

### Next steps (planned)
- Validate permeability reduction curve against experimental measurements
- Extend to full-domain geometry (see `full_domain/simulate_CuSO4_full_domain.py`)
- Benchmark MCMP Shan–Chen forces against published contact angle data
