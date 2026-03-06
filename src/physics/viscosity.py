import jax.numpy as jnp
from jax import jit

# -------------------------------------------------------------------
# [JAX-LaB Physics Module: CuSO4 Viscosity and Density]
# Price-Davenport (1980) + Laliberté-Cooper (2007) empirical model.
# -------------------------------------------------------------------


class CuSO4ViscositySimulator:
    """
    JAX-compatible empirical model for the dynamic viscosity and density
    of aqueous CuSO4 solutions.

    Valid Temperature Range: 20 °C – 100 °C.
    Concentration: up to the CuSO4·5H2O saturation limit.

    References
    ----------
    Price & Davenport (1980), Laliberté & Cooper (2007).
    """

    # Molar masses [g/mol]
    MW_CU        = 63.546
    MW_CUSO4     = 159.609
    MW_CUSO4_5H2O = 249.685

    # Price–Davenport dynamic viscosity constants
    PD_CONST     = 1592.0    # base constant
    PD_CU_LINEAR = 29.93     # linear-in-concentration coefficient
    PD_CU_SQRT   = 76.48     # sqrt-in-concentration coefficient (Falkenhagen term)
    PD_ACT_ENERGY = 1890.0   # effective activation energy divided by R  [K]

    # Price–Davenport solution density constants  [g/cm^3]
    DENS_BASE    = 1.01856
    DENS_CU      = 0.00238   # density increase per g/L of dissolved Cu
    DENS_TEMP    = 0.00059   # thermal-expansion coefficient  [per °C]

    # ----------------------------------------------------------------
    @staticmethod
    @jit
    def get_solubility_limit_gL_Cu(T_c: jnp.ndarray) -> jnp.ndarray:
        """
        Saturation limit of dissolved Cu [g/L] at temperature *T_c* [°C].

        Derived from a polynomial fit of CuSO4·5H2O solubility
        (CRC Handbook) converted to elemental copper concentration.
        """
        # Solubility of pentahydrate: g CuSO4·5H2O per 100 g water
        S_per_100g = 14.3 + 0.25 * T_c + 0.003 * (T_c ** 2)

        # Approximate solution density used for volumetric conversion
        rho_approx = 1.1 + 0.003 * T_c                          # g/cm^3

        # g/L of pentahydrate in solution
        conc_pentahydrate_gL = (S_per_100g / (100.0 + S_per_100g)) * rho_approx * 1000.0

        # Convert to g/L of elemental copper
        cu_gL = conc_pentahydrate_gL * (
            CuSO4ViscositySimulator.MW_CU / CuSO4ViscositySimulator.MW_CUSO4_5H2O
        )
        return cu_gL

    # ----------------------------------------------------------------
    @staticmethod
    @jit
    def calculate_dynamic_viscosity_cP(
        T_c: jnp.ndarray, Cu_gL: jnp.ndarray
    ) -> jnp.ndarray:
        """
        High-precision dynamic viscosity [mPa·s = cP] via the
        Price–Davenport empirical formulation.

        Parameters
        ----------
        T_c   : Temperature [°C]  – clipped to [20, 100] for model validity
        Cu_gL : Dissolved copper concentration [g/L]

        Returns
        -------
        Dynamic viscosity in mPa·s (cP).
        """
        # Clamp to the empirically validated range
        T_c  = jnp.clip(T_c, 20.0, 100.0)
        T_k  = T_c + 273.15

        # Enforce saturation boundary (no supersaturated viscosity extrapolation)
        sat_limit     = CuSO4ViscositySimulator.get_solubility_limit_gL_Cu(T_c)
        Cu_gL_bounded = jnp.minimum(Cu_gL, sat_limit)

        # Price–Davenport equation (Falkenhagen + Arrhenius)
        term_bracket = (
            CuSO4ViscositySimulator.PD_CONST
            + CuSO4ViscositySimulator.PD_CU_LINEAR * Cu_gL_bounded
            + CuSO4ViscositySimulator.PD_CU_SQRT   * jnp.sqrt(Cu_gL_bounded)
        )
        dynamic_viscosity = 1e-6 * term_bracket * jnp.exp(
            CuSO4ViscositySimulator.PD_ACT_ENERGY / T_k
        )
        return dynamic_viscosity  # [mPa·s = cP]

    # ----------------------------------------------------------------
    @staticmethod
    @jit
    def calculate_density_gcm3(
        T_c: jnp.ndarray, Cu_gL: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solution density [g/cm³] accounting for thermal expansion and
        dissolved copper concentration.
        """
        return (
            CuSO4ViscositySimulator.DENS_BASE
            + CuSO4ViscositySimulator.DENS_CU   * Cu_gL
            - CuSO4ViscositySimulator.DENS_TEMP * T_c
        )

    # ----------------------------------------------------------------
    @staticmethod
    @jit
    def calculate_kinematic_viscosity_cSt(
        T_c: jnp.ndarray, Cu_gL: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Kinematic viscosity [cSt = mm²/s] → ν = η / ρ.

        This is the primary scalar required for the LBM relaxation time:
            τ_f = 0.5 + ν_LB / c_s²

        Returns
        -------
        Kinematic viscosity in cSt (mm²/s).  Multiply by 1 × 10⁻⁶ to
        obtain SI units [m²/s].
        """
        dyn_visc = CuSO4ViscositySimulator.calculate_dynamic_viscosity_cP(T_c, Cu_gL)
        density  = CuSO4ViscositySimulator.calculate_density_gcm3(T_c, Cu_gL)
        return dyn_visc / density  # [cSt]
