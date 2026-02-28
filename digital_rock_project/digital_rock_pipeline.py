"""
=============================================================================
  3D Digital Rock Pipeline — JAX-LaB Ready
=============================================================================
  Phase 1 — Full Pipeline:
    Step 1 ▸ Data Ingestion      (TIFF stack / raw binary / synthetic)
    Step 2 ▸ Image Preprocessing  (Non-Local Means denoising)
    Step 3 ▸ Segmentation         (Otsu threshold + Watershed)
    Step 4 ▸ Simulation-Ready Mask (geometry_mask for LBM)
=============================================================================
  Dependencies:
    pip install numpy tifffile imageio scikit-image porespy matplotlib scipy PyWavelets
    PyWavelets is optional — a fallback sigma estimator is used if not installed.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import tifffile
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from scipy import ndimage

# scikit-image
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu, median
from skimage.restoration import denoise_nl_means

# estimate_sigma requires PyWavelets — use it if available, else fall back
try:
    from skimage.restoration import estimate_sigma
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False

def _estimate_sigma_fallback(image: np.ndarray) -> float:
    """
    Robust noise estimator based on the median absolute deviation of the
    high-frequency detail layer (derived from a Laplacian kernel).
    This is a well-known substitute when PyWavelets is not available.
    Reference: Donoho & Johnstone (1994).
    """
    from scipy.ndimage import laplace
    detail = laplace(image.astype(np.float64))
    return float(np.median(np.abs(detail)) / 0.6745)
from skimage.morphology import ball, closing, opening
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

# PoreSpy (optional — graceful fallback if not installed)
try:
    import porespy as ps
    PORESPY_AVAILABLE = True
    print("✔  PoreSpy detected — advanced SNOW partitioning available.")
except ImportError:
    PORESPY_AVAILABLE = False
    print("⚠   PoreSpy not found. Falling back to scikit-image watershed.")

if _PYWT_AVAILABLE:
    print("✔  PyWavelets detected — using estimate_sigma for NLM.")
else:
    print("⚠   PyWavelets not found. Using Laplacian MAD fallback for NLM sigma.")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these paths / settings before running)
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # ── Input ────────────────────────────────────────────────────────────────
    INPUT_MODE      = "raw_binary"       # "tiff_stack" | "raw_binary" | "synthetic"
    TIFF_DIR        = Path("./tiff_stack")          # folder of .tif slices
    RAW_PATH        = Path("./volume.raw")           # single raw binary file
    RAW_DTYPE       = np.uint16                      # dtype of raw file
    RAW_SHAPE       = (256, 256, 256)                # (Z, Y, X)

    # ── Synthetic rock (used when INPUT_MODE == "synthetic") ─────────────────
    SYNTH_SHAPE     = (64, 64, 64)        # voxels (Z, Y, X)
    SYNTH_POROSITY  = 0.35                # target porosity ≈ 35 %
    SYNTH_SEED      = 42

    # ── Preprocessing ─────────────────────────────────────────────────────────
    DENOISE_METHOD  = "nlm"              # "nlm" | "median"
    NLM_PATCH_SIZE  = 5                  # NLM patch size
    NLM_PATCH_DIST  = 6                  # NLM patch distance
    NLM_H_FACTOR    = 1.15               # NLM h = h_factor * sigma

    # ── Segmentation ──────────────────────────────────────────────────────────
    SEGM_METHOD     = "watershed"        # "otsu" | "watershed" | "porespy_snow"
    #                                      "multiphase_otsu" | "multiphase_manual"
    MIN_PORE_VOL    = 8                  # voxels — remove pores smaller than this

    # ── Sample type  (affects segmentation logic) ─────────────────────────────
    # "rock"        — standard 2-phase: grain vs pore
    # "beadpack"    — 3-phase microfluidic: air(pore) | glass bead(solid) | plastic(wall)
    SAMPLE_TYPE     = "beadpack"

    # ── Multi-phase thresholds (used when SAMPLE_TYPE = "beadpack") ───────────
    # Set both to None → auto-detect via multi-Otsu (recommended first try)
    # Override manually if auto-detect is wrong, e.g.:
    #   BEAD_THRESH_LOW  = 0.25   # below this = air / pore
    #   BEAD_THRESH_HIGH = 0.60   # above this = glass bead / solid
    #   voxels between LOW and HIGH = plastic wall (excluded from domain)
    BEAD_THRESH_LOW  = None      # None = auto via multi-Otsu
    BEAD_THRESH_HIGH = None      # None = auto via multi-Otsu

    # ── Output ────────────────────────────────────────────────────────────────
    OUTPUT_DIR      = Path("./output")
    SAVE_MASK_NPY   = True
    SAVE_MASK_TIFF  = True
    SAVE_FIGURES    = True

    # ── Crop UI ───────────────────────────────────────────────────────────────
    ENABLE_CROP_UI  = True     # show interactive crop window after ingestion


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
class DataIngestion:
    """Load micro-CT data into a 3D NumPy array (Z, Y, X)."""

    @staticmethod
    def _sanitise_volume(volume: np.ndarray, source: str) -> np.ndarray:
        """
        Ensure the loaded array is exactly 3-D (Z, Y, X).

        Micro-CT TIFFs come in many flavours:
          • Single multi-page TIFF  → tifffile returns (Z, Y, X)  ✔
          • Single multi-page TIFF  → sometimes (1, Z, Y, X)      ← squeeze
          • TIFF with colour channel → (Z, Y, X, C) or (Z, C, Y, X)
          • Stack of 2-D files      → after np.stack → (Z, Y, X)  ✔
        """
        raw_shape = volume.shape
        # Drop any length-1 leading/trailing dimensions
        volume = np.squeeze(volume)

        if volume.ndim == 2:
            # Only one slice — wrap into (1, Y, X)
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4:
            # Likely (Z, Y, X, C) colour or (C, Z, Y, X) — take mean over
            # the smallest axis as the channel axis
            channel_axis = int(np.argmin(volume.shape))
            volume = volume.mean(axis=channel_axis).astype(volume.dtype)
            print(f"[Ingestion] 4-D array detected {raw_shape} — "
                  f"collapsed axis {channel_axis} → {volume.shape}")
        elif volume.ndim != 3:
            raise ValueError(
                f"[Ingestion] Cannot handle {volume.ndim}-D array "
                f"from {source} (shape={raw_shape}). "
                "Expected a 3-D (Z, Y, X) volume.")

        print(f"[Ingestion] {source}  raw={raw_shape} → sanitised={volume.shape}  "
              f"dtype={volume.dtype}")
        return volume

    @staticmethod
    def load_tiff_stack(folder: Path) -> np.ndarray:
        """
        Load micro-CT data from a TIFF folder.

        Handles two layouts automatically:
          (a) A folder of 2-D per-slice TIFFs (slice_0001.tif …)
          (b) A single multi-page / multi-dimensional TIFF file
        """
        files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        if not files:
            raise FileNotFoundError(f"No TIFF files found in {folder}")

        if len(files) == 1:
            # Single multi-page TIFF — let tifffile read all pages at once
            volume = tifffile.imread(str(files[0]))
        else:
            # Stack of 2-D per-slice files
            slices = [tifffile.imread(str(f)) for f in files]
            volume = np.stack(slices, axis=0)

        return DataIngestion._sanitise_volume(volume, "TIFF stack")

    @staticmethod
    def load_raw_binary(path: Path, shape: tuple,
                        dtype: np.dtype) -> np.ndarray:
        """Load a flat binary volume file."""
        volume = np.fromfile(str(path), dtype=dtype).reshape(shape)
        return DataIngestion._sanitise_volume(volume, "raw binary")

    @staticmethod
    def generate_synthetic_rock(shape: tuple = (64, 64, 64),
                                porosity: float = 0.35,
                                seed: int = 42) -> np.ndarray:
        """
        Generate a realistic synthetic grayscale rock volume using
        overlapping Gaussian blobs as grain proxies.
        Returns a uint16 grayscale volume.
        """
        rng  = np.random.default_rng(seed)
        vol  = np.zeros(shape, dtype=np.float32)

        n_grains   = int(np.prod(shape) * (1 - porosity) / 500)
        grain_radii = rng.integers(4, 10, size=n_grains)

        for r in grain_radii:
            cx = rng.integers(0, shape[2])
            cy = rng.integers(0, shape[1])
            cz = rng.integers(0, shape[0])
            z, y, x = np.ogrid[
                max(0, cz-r):min(shape[0], cz+r),
                max(0, cy-r):min(shape[1], cy+r),
                max(0, cx-r):min(shape[2], cx+r)
            ]
            dist = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
            region = tuple([
                slice(max(0, cz-r), min(shape[0], cz+r)),
                slice(max(0, cy-r), min(shape[1], cy+r)),
                slice(max(0, cx-r), min(shape[2], cx+r)),
            ])
            mask = dist < r
            vol[region][mask] += 1.0 - dist[mask] / r

        # Add noise to mimic scanner variability
        noise = rng.normal(0, 0.05, shape).astype(np.float32)
        vol   = np.clip(vol + noise, 0, None)

        # Normalise to uint16
        vol   = (vol / vol.max() * 65535).astype(np.uint16)
        print(f"[Ingestion] Synthetic rock     shape={vol.shape}  "
              f"dtype={vol.dtype}")
        return vol

    @staticmethod
    def load(cfg: Config) -> np.ndarray:
        if cfg.INPUT_MODE == "tiff_stack":
            return DataIngestion.load_tiff_stack(cfg.TIFF_DIR)
        elif cfg.INPUT_MODE == "raw_binary":
            return DataIngestion.load_raw_binary(cfg.RAW_PATH,
                                                  cfg.RAW_SHAPE,
                                                  cfg.RAW_DTYPE)
        elif cfg.INPUT_MODE == "synthetic":
            return DataIngestion.generate_synthetic_rock(
                cfg.SYNTH_SHAPE, cfg.SYNTH_POROSITY, cfg.SYNTH_SEED)
        else:
            raise ValueError(f"Unknown INPUT_MODE: {cfg.INPUT_MODE}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — IMAGE PREPROCESSING (DENOISING)
# ─────────────────────────────────────────────────────────────────────────────
class Preprocessing:
    """
    Remove scan noise while preserving grain-boundary sharpness —
    critical for accurate LBM bounce-back boundary conditions.
    """

    @staticmethod
    def denoise_nlm(volume: np.ndarray,
                    patch_size: int  = 5,
                    patch_dist: int  = 6,
                    h_factor: float  = 1.15) -> np.ndarray:
        """
        Non-Local Means denoising (slice-by-slice to manage memory).
        Each 2-D slice is denoised independently, which is a common
        approximation for volumetric micro-CT data.
        """
        print("[Preprocess] Applying Non-Local Means denoising …")
        vol_float  = img_as_float(volume)
        denoised   = np.empty_like(vol_float)

        for z_idx in range(vol_float.shape[0]):
            slc   = vol_float[z_idx]
            if _PYWT_AVAILABLE:
                sigma = float(estimate_sigma(slc))
            else:
                sigma = _estimate_sigma_fallback(slc)
            h     = h_factor * sigma
            denoised[z_idx] = denoise_nl_means(
                slc,
                h            = h,
                patch_size   = patch_size,
                patch_distance = patch_dist,
                fast_mode    = True,
            )
            if z_idx % 10 == 0:
                print(f"  slice {z_idx:>4d}/{vol_float.shape[0]}  "
                      f"σ={sigma:.4f}  h={h:.4f}")

        print("[Preprocess] NLM denoising complete.")
        return denoised   # float64 in [0, 1]

    @staticmethod
    def denoise_median(volume: np.ndarray,
                       footprint_radius: int = 1) -> np.ndarray:
        """
        Faster but less edge-preserving alternative: 3-D median filter.
        """
        print("[Preprocess] Applying Median filter …")
        fp      = ball(footprint_radius)
        vol_u8  = img_as_ubyte(
            (volume.astype(np.float64) / volume.max()).astype(np.float32))
        result  = median(vol_u8, footprint=fp)
        return result.astype(np.float64) / 255.0

    @staticmethod
    def run(volume: np.ndarray, cfg: Config) -> np.ndarray:
        if cfg.DENOISE_METHOD == "nlm":
            return Preprocessing.denoise_nlm(
                volume, cfg.NLM_PATCH_SIZE,
                cfg.NLM_PATCH_DIST, cfg.NLM_H_FACTOR)
        elif cfg.DENOISE_METHOD == "median":
            return Preprocessing.denoise_median(volume)
        else:
            raise ValueError(f"Unknown DENOISE_METHOD: {cfg.DENOISE_METHOD}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class Segmentation:
    """
    Convert the continuous grayscale volume into a binary phase map:
        True  →  Solid (grain)
        False →  Fluid (pore)
    """

    @staticmethod
    def otsu_threshold(vol_float: np.ndarray) -> np.ndarray:
        """Simple global Otsu threshold — works well for bimodal histograms."""
        thresh   = threshold_otsu(vol_float)
        binary   = vol_float > thresh
        print(f"[Segmentation] Otsu threshold = {thresh:.4f}  "
              f"solid fraction = {binary.mean():.3f}")
        return binary

    @staticmethod
    def watershed_segmentation(vol_float: np.ndarray,
                                min_pore_vol: int = 8) -> np.ndarray:
        """
        Marker-based watershed on the Distance Transform.
        Steps:
          1. Otsu binary mask
          2. Distance transform of the pore space
          3. Local maxima → seed markers  (adaptive min_distance fallback)
          4. Watershed to separate touching grains / define pore throats
          5. Remove tiny isolated pore clusters (noise)
        """
        print("[Segmentation] Running Watershed segmentation …")

        # 1. Initial binary mask
        thresh  = threshold_otsu(vol_float)
        solid   = vol_float > thresh          # True = grain
        pore    = ~solid

        porosity = pore.mean()
        print(f"[Segmentation] Otsu threshold = {thresh:.4f}  "
              f"porosity = {porosity:.3f}  solid = {solid.mean():.3f}")

        if porosity < 0.01:
            print("[Segmentation] ⚠  Very low porosity detected — "
                  "returning Otsu mask directly (no pore seeds to watershed).")
            return solid

        # 2. Distance transform
        dist = ndimage.distance_transform_edt(pore)

        # 3. Local maxima as seeds — try progressively smaller footprints
        #    until at least one seed is found
        coords = np.empty((0, vol_float.ndim), dtype=int)
        for radius in (3, 2, 1):
            coords = peak_local_max(
                dist,
                min_distance = radius,
                footprint    = ball(radius),
                labels       = pore.astype(np.uint8),
            )
            if len(coords) > 0:
                print(f"[Segmentation] Found {len(coords)} pore seeds "
                      f"(footprint radius={radius})")
                break

        if len(coords) == 0:
            # Absolute fallback — label every pore voxel as its own seed
            print("[Segmentation] ⚠  No local maxima found — "
                  "falling back to Otsu-only segmentation.")
            return solid

        markers = np.zeros_like(dist, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        markers = ndimage.label(markers)[0]   # merge touching seed points

        # 4. Watershed — inverted distance so peaks become valleys
        labels = watershed(-dist, markers, mask=pore)

        # 5. Remove small pores
        labeled_pore = label(labels > 0)
        props        = regionprops(labeled_pore)
        keep         = np.zeros_like(labeled_pore, dtype=bool)
        for p in props:
            if p.area >= min_pore_vol:
                keep[labeled_pore == p.label] = True

        # Final mask: solid = True, fluid = False
        result = ~keep        # invert: pore→fluid (False), grain→solid (True)
        # Fill back original solid voxels
        result = result | solid

        print(f"[Segmentation] Watershed complete — "
              f"solid fraction = {result.mean():.3f}  "
              f"porosity = {1 - result.mean():.3f}")
        return result

    @staticmethod
    def porespy_snow(vol_float: np.ndarray) -> np.ndarray:
        """
        SNOW algorithm from PoreSpy — recommended for pore-throat accuracy.
        Returns the solid mask (True = grain).
        """
        print("[Segmentation] Running PoreSpy SNOW partitioning …")
        thresh   = threshold_otsu(vol_float)
        solid    = vol_float > thresh
        pore     = ~solid

        snow_out = ps.filters.snow_partitioning(pore.astype(float))
        pore_seg = snow_out.regions > 0

        result = ~pore_seg
        print(f"[Segmentation] SNOW complete — "
              f"solid fraction = {result.mean():.3f}  "
              f"porosity = {1 - result.mean():.3f}")
        return result

    @staticmethod
    def multiphase_otsu(vol_float: np.ndarray) -> dict:
        """
        3-class segmentation for microfluidic bead-pack using
        multi-threshold Otsu (finds 2 thresholds automatically).

        Returns a dict with:
            'solid'    — bool mask  (glass beads)
            'fluid'    — bool mask  (air / pore space)
            'wall'     — bool mask  (plastic cell body — excluded from domain)
            'thresh_low', 'thresh_high'  — the two thresholds found
        """
        from skimage.filters import threshold_multiotsu
        print("[Segmentation] Running Multi-Otsu (3-class beadpack) …")

        thresholds = threshold_multiotsu(vol_float, classes=3)
        t_low, t_high = float(thresholds[0]), float(thresholds[1])
        print(f"[Segmentation] Multi-Otsu thresholds:  "
              f"low={t_low:.4f}  high={t_high:.4f}")

        fluid  = vol_float <= t_low                          # black  = air
        wall   = (vol_float > t_low) & (vol_float <= t_high) # gray   = plastic
        solid  = vol_float > t_high                          # white  = glass bead

        print(f"[Segmentation]   fluid (air)   = {fluid.mean()*100:.1f} %")
        print(f"[Segmentation]   wall (plastic)= {wall.mean()*100:.1f} %")
        print(f"[Segmentation]   solid (bead)  = {solid.mean()*100:.1f} %")

        return dict(solid=solid, fluid=fluid, wall=wall,
                    thresh_low=t_low, thresh_high=t_high)

    @staticmethod
    def multiphase_manual(vol_float: np.ndarray,
                          thresh_low: float,
                          thresh_high: float) -> dict:
        """Same as multiphase_otsu but with user-supplied thresholds."""
        print(f"[Segmentation] Manual 3-class thresholds:  "
              f"low={thresh_low:.4f}  high={thresh_high:.4f}")
        fluid = vol_float <= thresh_low
        wall  = (vol_float > thresh_low) & (vol_float <= thresh_high)
        solid = vol_float > thresh_high
        print(f"[Segmentation]   fluid={fluid.mean()*100:.1f} %  "
              f"wall={wall.mean()*100:.1f} %  "
              f"solid={solid.mean()*100:.1f} %")
        return dict(solid=solid, fluid=fluid, wall=wall,
                    thresh_low=thresh_low, thresh_high=thresh_high)

    @staticmethod
    def beadpack_to_mask(phases: dict, min_pore_vol: int = 8) -> np.ndarray:
        """
        Convert 3-phase classification into a JAX-LaB geometry_mask.

        Convention (inside the cropped channel only):
            1  =  solid (glass bead — bounce-back)
            0  =  fluid (air / pore — LBM streaming)

        The 'wall' (plastic) phase is treated as solid because after
        cropping to the inner channel it should be minimal / absent.
        If significant wall voxels remain post-crop, consider re-cropping.
        """
        solid_mask = phases["solid"] | phases["wall"]   # plastic → solid wall

        # Remove isolated tiny pore clusters (scan noise)
        pore       = ~solid_mask
        labeled    = label(pore)
        props      = regionprops(labeled)
        keep       = np.zeros_like(pore, dtype=bool)
        for p in props:
            if p.area >= min_pore_vol:
                keep[labeled == p.label] = True
        solid_mask = ~keep

        porosity = (~solid_mask).mean()
        print(f"[Segmentation] Beadpack mask complete — "
              f"porosity = {porosity*100:.2f} %  "
              f"solid = {solid_mask.mean()*100:.2f} %")
        return solid_mask

    @staticmethod
    def run(vol_float: np.ndarray, cfg: Config) -> np.ndarray:
        # ── Beadpack / microfluidic path ─────────────────────────────────────
        if cfg.SAMPLE_TYPE == "beadpack":
            if (cfg.BEAD_THRESH_LOW is not None
                    and cfg.BEAD_THRESH_HIGH is not None):
                phases = Segmentation.multiphase_manual(
                    vol_float,
                    cfg.BEAD_THRESH_LOW,
                    cfg.BEAD_THRESH_HIGH)
            else:
                phases = Segmentation.multiphase_otsu(vol_float)
            return Segmentation.beadpack_to_mask(phases, cfg.MIN_PORE_VOL)

        # ── Standard rock path ───────────────────────────────────────────────
        if cfg.SEGM_METHOD == "otsu":
            return Segmentation.otsu_threshold(vol_float)
        elif cfg.SEGM_METHOD == "watershed":
            return Segmentation.watershed_segmentation(
                vol_float, cfg.MIN_PORE_VOL)
        elif cfg.SEGM_METHOD == "porespy_snow":
            if not PORESPY_AVAILABLE:
                print("[Segmentation] PoreSpy not installed — "
                      "falling back to watershed.")
                return Segmentation.watershed_segmentation(
                    vol_float, cfg.MIN_PORE_VOL)
            return Segmentation.porespy_snow(vol_float)
        else:
            raise ValueError(f"Unknown SEGM_METHOD: {cfg.SEGM_METHOD}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — SIMULATION-READY MASK
# ─────────────────────────────────────────────────────────────────────────────
class SimulationMask:
    """
    Produce the final geometry_mask in JAX-LaB convention:
        1 (True)  →  Solid obstacle (bounce-back node)
        0 (False) →  Fluid pore     (LBM streaming node)
    """

    @staticmethod
    def build(binary_solid: np.ndarray,
              apply_closing: bool = True,
              closing_radius: int = 1) -> np.ndarray:
        """
        Optional morphological closing to seal sub-voxel cracks in
        grain boundaries before exporting.
        """
        mask = binary_solid.astype(bool)

        if apply_closing:
            print(f"[Mask] Morphological closing (r={closing_radius}) …")
            mask = closing(mask, ball(closing_radius))

        mask = mask.astype(np.uint8)      # 0 = fluid, 1 = solid (JAX-LaB)
        print(f"[Mask] geometry_mask  shape={mask.shape}  "
              f"dtype={mask.dtype}  "
              f"porosity={1 - mask.mean():.4f}")
        return mask

    @staticmethod
    def compute_statistics(mask: np.ndarray) -> dict:
        solid   = mask.astype(bool)
        pore    = ~solid
        stats   = {
            "shape"         : mask.shape,
            "total_voxels"  : int(mask.size),
            "solid_voxels"  : int(solid.sum()),
            "pore_voxels"   : int(pore.sum()),
            "porosity"      : float(pore.sum() / mask.size),
            "solid_fraction": float(solid.sum() / mask.size),
        }
        # Pore-size distribution via distance transform
        dist            = ndimage.distance_transform_edt(pore)
        pore_radii      = dist[pore]
        if pore_radii.size > 0:
            stats["mean_pore_radius_vox"]   = float(pore_radii.mean())
            stats["max_pore_radius_vox"]    = float(pore_radii.max())
        else:
            stats["mean_pore_radius_vox"]   = 0.0
            stats["max_pore_radius_vox"]    = 0.0
        return stats

    @staticmethod
    def save(mask: np.ndarray, cfg: Config) -> None:
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if cfg.SAVE_MASK_NPY:
            npy_path = cfg.OUTPUT_DIR / "geometry_mask.npy"
            np.save(str(npy_path), mask)
            print(f"[Save] NumPy array  → {npy_path}")
        if cfg.SAVE_MASK_TIFF:
            tiff_path = cfg.OUTPUT_DIR / "geometry_mask.tiff"
            tifffile.imwrite(str(tiff_path), mask)
            print(f"[Save] TIFF volume  → {tiff_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
class Visualiser:
    """Side-by-side diagnostic plots at every pipeline stage."""

    @staticmethod
    def _mid(arr: np.ndarray) -> int:
        return arr.shape[0] // 2

    @staticmethod
    def plot_pipeline(raw: np.ndarray,
                      denoised: np.ndarray,
                      binary: np.ndarray,
                      mask: np.ndarray,
                      stats: dict,
                      save_path: Path | None = None) -> None:

        mid_raw  = Visualiser._mid(raw)
        mid_den  = Visualiser._mid(denoised)
        mid_bin  = Visualiser._mid(binary)
        mid_msk  = Visualiser._mid(mask)

        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor("#0d0d0d")
        gs  = gridspec.GridSpec(3, 4, figure=fig,
                                hspace=0.40, wspace=0.30)

        # ── Colour maps ──────────────────────────────────────────────────────
        GREY   = "gray"
        BINARY = plt.cm.RdYlBu_r
        MASK_C = plt.cm.coolwarm

        title_kw  = dict(color="white", fontsize=10, pad=6)
        label_kw  = dict(color="#aaaaaa", fontsize=8)

        def ax_img(gs_pos, data, cmap, title, vmin=None, vmax=None):
            ax = fig.add_subplot(gs_pos)
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(title, **title_kw)
            ax.axis("off")
            ax.set_facecolor("#0d0d0d")
            return ax

        # ── Row 0: XY slices ─────────────────────────────────────────────────
        ax_img(gs[0, 0], raw[mid_raw],      GREY,   "① Raw  (XY)")
        ax_img(gs[0, 1], denoised[mid_den], GREY,
               "② Denoised — NLM  (XY)", 0, 1)
        ax_img(gs[0, 2], binary[mid_bin].astype(float), BINARY,
               "③ Segmented  (XY)", 0, 1)
        ax_img(gs[0, 3], mask[mid_msk],     MASK_C,
               "④ geometry_mask  (XY)\n  ■ Solid = 1   □ Fluid = 0", 0, 1)

        # ── Row 1: XZ slices (orthogonal view) ───────────────────────────────
        mid_y = raw.shape[1] // 2
        ax_img(gs[1, 0], raw[:, mid_y, :],        GREY,   "① Raw  (XZ)")
        ax_img(gs[1, 1], denoised[:, mid_y, :],   GREY,
               "② Denoised  (XZ)", 0, 1)
        ax_img(gs[1, 2], binary[:, mid_y, :].astype(float), BINARY,
               "③ Segmented  (XZ)", 0, 1)
        ax_img(gs[1, 3], mask[:, mid_y, :],        MASK_C,
               "④ geometry_mask  (XZ)", 0, 1)

        # ── Row 2: Histogram + Statistics ────────────────────────────────────
        ax_hist = fig.add_subplot(gs[2, :2])
        ax_hist.set_facecolor("#1a1a1a")
        flat_raw = raw.ravel().astype(np.float64)
        flat_raw = (flat_raw - flat_raw.min()) / (flat_raw.max()
                                                   - flat_raw.min() + 1e-9)
        ax_hist.hist(flat_raw, bins=256, color="#4fc3f7",
                     alpha=0.7, label="Raw (norm.)")
        ax_hist.hist(denoised.ravel(), bins=256, color="#81c784",
                     alpha=0.7, label="Denoised")
        thresh_otsu = threshold_otsu(denoised)
        ax_hist.axvline(thresh_otsu, color="#ff7043", linewidth=1.5,
                        linestyle="--", label=f"Otsu = {thresh_otsu:.3f}")
        ax_hist.set_title("Grayscale Histogram — Raw vs Denoised",
                           **title_kw)
        ax_hist.set_xlabel("Normalised Intensity", **label_kw)
        ax_hist.set_ylabel("Voxel Count", **label_kw)
        ax_hist.tick_params(colors="#aaaaaa")
        for sp in ax_hist.spines.values():
            sp.set_edgecolor("#444444")
        ax_hist.legend(fontsize=8, labelcolor="white",
                        facecolor="#2a2a2a", edgecolor="#555555")

        # Statistics text box
        ax_stat = fig.add_subplot(gs[2, 2:])
        ax_stat.set_facecolor("#1a1a1a")
        ax_stat.axis("off")
        stat_lines = [
            "📦  Volume Statistics (geometry_mask)",
            "─" * 38,
            f"  Shape            {stats['shape']}",
            f"  Total voxels     {stats['total_voxels']:,}",
            f"  Solid voxels     {stats['solid_voxels']:,}",
            f"  Pore  voxels     {stats['pore_voxels']:,}",
            f"  Porosity         {stats['porosity']:.4f}  "
            f"({stats['porosity']*100:.2f} %)",
            f"  Solid fraction   {stats['solid_fraction']:.4f}",
            f"  Mean pore radius {stats['mean_pore_radius_vox']:.2f} vox",
            f"  Max  pore radius {stats['max_pore_radius_vox']:.2f} vox",
            "─" * 38,
            "  JAX-LaB convention:",
            "  1 = Solid (bounce-back) | 0 = Fluid",
        ]
        ax_stat.text(0.03, 0.95, "\n".join(stat_lines),
                     transform=ax_stat.transAxes,
                     va="top", ha="left",
                     fontsize=9, color="white",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#2a2a2a",
                               edgecolor="#4fc3f7", alpha=0.9))

        fig.suptitle("3D Digital Rock Pipeline  —  JAX-LaB Ready",
                     color="white", fontsize=14, fontweight="bold", y=0.98)

        plt.savefig(str(save_path), dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Visualise] Figure saved → {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CROP UI
# ─────────────────────────────────────────────────────────────────────────────
class CropUI:
    """
    Interactive Matplotlib UI for cropping non-rock border regions.

    Shows three orthogonal views (XY / XZ / YZ) simultaneously.
    Six sliders control the crop box:  Z_min, Z_max, Y_min, Y_max, X_min, X_max
    A live red rectangle overlays every view to show the current crop region.
    Pressing [✔ Confirm Crop] returns the cropped volume and exits.
    Pressing [✖ Skip Crop]   returns the original volume unchanged.

    Usage
    -----
    cropped = CropUI.run(raw_volume)
    """

    @staticmethod
    def run(volume: np.ndarray) -> np.ndarray:
        """Launch the crop UI.  Returns the (possibly cropped) volume."""
        from matplotlib.widgets import Slider, Button
        import matplotlib.patches as mpatches

        Z, Y, X = volume.shape
        result   = {"vol": volume, "cropped": False}

        # ── Normalise for display ────────────────────────────────────────────
        vmin = float(volume.min())
        vmax = float(volume.max())
        def _norm(arr):
            return (arr.astype(np.float32) - vmin) / (vmax - vmin + 1e-9)

        # ── Figure layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 13), facecolor="#0d0d0d")
        fig.suptitle(
            "✂  Volume Crop Tool  —  drag sliders to remove non-rock borders",
            color="white", fontsize=13, fontweight="bold", y=0.98)

        # 3 image axes  +  6 slider axes  +  2 button axes
        gs_top = gridspec.GridSpec(
            1, 3, figure=fig,
            left=0.04, right=0.98, top=0.90, bottom=0.44,
            wspace=0.08)
        gs_mid = gridspec.GridSpec(
            6, 2, figure=fig,
            left=0.08, right=0.92, top=0.40, bottom=0.08,
            hspace=0.55, wspace=0.40)

        ax_xy = fig.add_subplot(gs_top[0, 0])
        ax_xz = fig.add_subplot(gs_top[0, 1])
        ax_yz = fig.add_subplot(gs_top[0, 2])

        for ax, title in zip([ax_xy, ax_xz, ax_yz],
                              ["XY  (top view — Z slice)",
                               "XZ  (front view — Y slice)",
                               "YZ  (side view — X slice)"]):
            ax.set_facecolor("#1a1a1a")
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="#666666", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

        # ── Slider definitions: (label, min, max, init_min, init_max) ────────
        SLIDER_COLOR  = "#1e3a5f"
        ACTIVE_COLOR  = "#4fc3f7"

        sliders = {}
        slider_specs = [
            ("Z min", 0, Z - 1, 0),
            ("Z max", 1, Z,     Z),
            ("Y min", 0, Y - 1, 0),
            ("Y max", 1, Y,     Y),
            ("X min", 0, X - 1, 0),
            ("X max", 1, X,     X),
        ]
        for i, (lbl, lo, hi, init) in enumerate(slider_specs):
            row, col = i // 2, i % 2
            ax_s = fig.add_subplot(gs_mid[row, col])
            ax_s.set_facecolor("#0d0d0d")
            sl = Slider(ax_s, lbl, lo, hi,
                        valinit=init, valstep=1,
                        color=SLIDER_COLOR)
            sl.label.set_color("white")
            sl.valtext.set_color(ACTIVE_COLOR)
            sl.track.set_facecolor("#2a2a2a")
            sliders[lbl] = sl

        # ── Button axes ───────────────────────────────────────────────────────
        ax_btn_confirm = fig.add_axes([0.30, 0.01, 0.18, 0.045])
        ax_btn_skip    = fig.add_axes([0.52, 0.01, 0.18, 0.045])
        btn_confirm = Button(ax_btn_confirm, "✔  Confirm Crop",
                             color="#1b5e20", hovercolor="#2e7d32")
        btn_skip    = Button(ax_btn_skip,    "✖  Skip  (use full volume)",
                             color="#4a1010", hovercolor="#7f1d1d")
        for btn in (btn_confirm, btn_skip):
            btn.label.set_color("white")
            btn.label.set_fontsize(9)

        # ── Draw helper ───────────────────────────────────────────────────────
        def _get_crop():
            z0 = int(sliders["Z min"].val)
            z1 = int(sliders["Z max"].val)
            y0 = int(sliders["Y min"].val)
            y1 = int(sliders["Y max"].val)
            x0 = int(sliders["X min"].val)
            x1 = int(sliders["X max"].val)
            # Enforce min < max with at least 1 voxel gap
            z1 = max(z1, z0 + 1)
            y1 = max(y1, y0 + 1)
            x1 = max(x1, x0 + 1)
            return z0, z1, y0, y1, x0, x1

        rect_handles = [None, None, None]   # XY, XZ, YZ

        def _redraw(_=None):
            z0, z1, y0, y1, x0, x1 = _get_crop()
            z_mid = (z0 + z1) // 2
            y_mid = (y0 + y1) // 2
            x_mid = (x0 + x1) // 2

            # XY view: imshow(volume[z_mid], origin="upper")  → cols=X, rows=Y
            ax_xy.cla()
            ax_xy.imshow(_norm(volume[z_mid]), cmap="gray",
                          origin="upper", aspect="auto",
                          interpolation="nearest")
            ax_xy.set_title(f"XY  — Z slice {z_mid}  "
                             f"(crop Z {z0}:{z1})",
                             color="white", fontsize=8)
            # Red rectangle: x=X_range, y=Y_range
            rect_xy = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor="#ff4444",
                facecolor="none", linestyle="--")
            ax_xy.add_patch(rect_xy)
            ax_xy.tick_params(colors="#666666", labelsize=7)

            # XZ view: imshow(volume[:, y_mid, :]) → cols=X, rows=Z
            ax_xz.cla()
            ax_xz.imshow(_norm(volume[:, y_mid, :]), cmap="gray",
                          origin="upper", aspect="auto",
                          interpolation="nearest")
            ax_xz.set_title(f"XZ  — Y slice {y_mid}  "
                             f"(crop Y {y0}:{y1})",
                             color="white", fontsize=8)
            rect_xz = mpatches.Rectangle(
                (x0, z0), x1 - x0, z1 - z0,
                linewidth=1.5, edgecolor="#ff4444",
                facecolor="none", linestyle="--")
            ax_xz.add_patch(rect_xz)
            ax_xz.tick_params(colors="#666666", labelsize=7)

            # YZ view: imshow(volume[:, :, x_mid]) → cols=Y, rows=Z
            ax_yz.cla()
            ax_yz.imshow(_norm(volume[:, :, x_mid]), cmap="gray",
                          origin="upper", aspect="auto",
                          interpolation="nearest")
            ax_yz.set_title(f"YZ  — X slice {x_mid}  "
                             f"(crop X {x0}:{x1})",
                             color="white", fontsize=8)
            rect_yz = mpatches.Rectangle(
                (y0, z0), y1 - y0, z1 - z0,
                linewidth=1.5, edgecolor="#ff4444",
                facecolor="none", linestyle="--")
            ax_yz.add_patch(rect_yz)
            ax_yz.tick_params(colors="#666666", labelsize=7)

            # Info label
            dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
            fig.texts = [t for t in fig.texts
                         if not getattr(t, "_crop_info", False)]
            info = fig.text(
                0.50, 0.425,
                f"Crop region:  Z [{z0}:{z1}]  Y [{y0}:{y1}]  X [{x0}:{x1}]"
                f"   →   {dz} × {dy} × {dx} voxels  "
                f"({dz * dy * dx / volume.size * 100:.1f} % of original)",
                ha="center", va="top", color="#4fc3f7",
                fontsize=9, fontfamily="monospace")
            info._crop_info = True
            fig.canvas.draw_idle()

        # Hook all sliders
        for sl in sliders.values():
            sl.on_changed(_redraw)

        # ── Button callbacks ──────────────────────────────────────────────────
        def _confirm(event):
            z0, z1, y0, y1, x0, x1 = _get_crop()
            cropped = volume[z0:z1, y0:y1, x0:x1].copy()
            print(f"[Crop] ✔ Confirmed  {volume.shape} → {cropped.shape}  "
                  f"(Z {z0}:{z1}  Y {y0}:{y1}  X {x0}:{x1})")
            result["vol"]     = cropped
            result["cropped"] = True
            plt.close(fig)

        def _skip(event):
            print(f"[Crop] ✖ Skipped — using full volume {volume.shape}")
            plt.close(fig)

        btn_confirm.on_clicked(_confirm)
        btn_skip.on_clicked(_skip)

        # Initial draw
        _redraw()
        plt.show(block=True)   # block until window is closed

        return result["vol"]


# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLD SELECTOR UI  (beadpack / 3-phase)
# ─────────────────────────────────────────────────────────────────────────────
class ThresholdUI:
    """
    Interactive histogram tool for beadpack 3-phase threshold selection.
    Shows the grayscale histogram with two draggable threshold lines.
    Three coloured regions visualise: Fluid (blue) | Wall (gray) | Solid (red).
    A mid-Z slice preview updates live to show the segmentation result.

    Returns (thresh_low, thresh_high) — or (None, None) for auto Multi-Otsu.
    """

    @staticmethod
    def run(vol_float: np.ndarray) -> tuple:
        from matplotlib.widgets import Slider, Button

        result = {"low": None, "high": None, "use_auto": True}

        # Auto-compute multi-Otsu as a sensible starting point
        try:
            from skimage.filters import threshold_multiotsu
            t0, t1 = [float(t) for t in threshold_multiotsu(vol_float, classes=3)]
        except Exception:
            t0, t1 = 0.25, 0.65

        fig, axes = plt.subplots(
            1, 2, figsize=(18, 7), facecolor="#0d0d0d")
        fig.suptitle(
            "🔬  Beadpack Threshold Tool  —  "
            "set air↔plastic (low) and plastic↔bead (high) boundaries",
            color="white", fontsize=12, fontweight="bold")
        fig.subplots_adjust(bottom=0.28, wspace=0.12)

        ax_hist, ax_prev = axes
        for ax in axes:
            ax.set_facecolor("#1a1a1a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")
            ax.tick_params(colors="#aaaaaa")

        # ── Histogram ────────────────────────────────────────────────────────
        flat    = vol_float.ravel()
        n_bins  = 512
        counts, bin_edges = np.histogram(flat, bins=n_bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax_hist.bar(bin_centres, counts, width=bin_edges[1] - bin_edges[0],
                    color="#555555", alpha=0.8)
        ax_hist.set_title("Grayscale Histogram — drag sliders to set thresholds",
                           color="white", fontsize=9)
        ax_hist.set_xlabel("Normalised Intensity", color="#aaaaaa", fontsize=8)
        ax_hist.set_ylabel("Voxel Count",          color="#aaaaaa", fontsize=8)

        # Coloured fill regions + threshold lines (stored as lists for mutability)
        fills = [
            ax_hist.axvspan(0,  t0, alpha=0.25, color="#1565c0", label="Air / pore (fluid)"),
            ax_hist.axvspan(t0, t1, alpha=0.25, color="#78909c", label="Plastic wall"),
            ax_hist.axvspan(t1,  1, alpha=0.25, color="#b71c1c", label="Glass bead (solid)"),
        ]
        vl_low  = ax_hist.axvline(t0, color="#4fc3f7", linewidth=2,
                                   linestyle="-", label=f"T_low  = {t0:.3f}")
        vl_high = ax_hist.axvline(t1, color="#ef5350", linewidth=2,
                                   linestyle="-", label=f"T_high = {t1:.3f}")
        ax_hist.legend(fontsize=8, facecolor="#2a2a2a",
                       edgecolor="#555555", labelcolor="white")

        # ── Slice preview ─────────────────────────────────────────────────────
        mid_z = vol_float.shape[0] // 2
        slc   = vol_float[mid_z]

        def _make_rgb(low, high):
            rgb = np.zeros((*slc.shape, 3), dtype=np.float32)
            rgb[slc <= low]                        = [0.08, 0.39, 0.75]  # blue  = air
            rgb[(slc > low) & (slc <= high)]       = [0.47, 0.57, 0.60]  # gray  = plastic
            rgb[slc > high]                        = [0.92, 0.26, 0.21]  # red   = bead
            return rgb

        img_handle = ax_prev.imshow(_make_rgb(t0, t1), aspect="auto",
                                     interpolation="nearest", origin="upper")
        ax_prev.set_title(
            f"Segmentation preview — Z slice {mid_z}\n"
            "🔵 Air (fluid)   ◻ Plastic (wall)   🔴 Glass bead (solid)",
            color="white", fontsize=8)

        # ── Sliders ───────────────────────────────────────────────────────────
        ax_sl_low  = fig.add_axes([0.10, 0.15, 0.80, 0.030])
        ax_sl_high = fig.add_axes([0.10, 0.09, 0.80, 0.030])
        for ax in (ax_sl_low, ax_sl_high):
            ax.set_facecolor("#0d0d0d")

        sl_low  = Slider(ax_sl_low,  "T low  (air | plastic)",
                         0.0, 1.0, valinit=t0, valstep=0.001, color="#1e3a5f")
        sl_high = Slider(ax_sl_high, "T high (plastic | bead)",
                         0.0, 1.0, valinit=t1, valstep=0.001, color="#5f1e1e")
        for sl in (sl_low, sl_high):
            sl.label.set_color("white")
            sl.valtext.set_color("#4fc3f7")

        def _update(_=None):
            low  = float(sl_low.val)
            high = max(float(sl_high.val), low + 0.001)
            # Redraw fill regions
            for fill in fills:
                fill.remove()
            fills.clear()
            fills.append(ax_hist.axvspan(0,   low,  alpha=0.25, color="#1565c0"))
            fills.append(ax_hist.axvspan(low,  high, alpha=0.25, color="#78909c"))
            fills.append(ax_hist.axvspan(high, 1,    alpha=0.25, color="#b71c1c"))
            vl_low.set_xdata([low, low])
            vl_high.set_xdata([high, high])
            img_handle.set_data(_make_rgb(low, high))
            fig.canvas.draw_idle()

        sl_low.on_changed(_update)
        sl_high.on_changed(_update)

        # ── Buttons ───────────────────────────────────────────────────────────
        ax_btn_auto   = fig.add_axes([0.20, 0.02, 0.18, 0.045])
        ax_btn_manual = fig.add_axes([0.41, 0.02, 0.18, 0.045])
        ax_btn_skip   = fig.add_axes([0.62, 0.02, 0.18, 0.045])

        btn_auto   = Button(ax_btn_auto,   "🤖  Auto Multi-Otsu",
                            color="#1b3a1b", hovercolor="#2e7d32")
        btn_manual = Button(ax_btn_manual, "✔  Use These Thresholds",
                            color="#1b3560", hovercolor="#1565c0")
        btn_skip   = Button(ax_btn_skip,   "✖  Skip (2-phase Otsu)",
                            color="#4a1010", hovercolor="#7f1d1d")
        for btn in (btn_auto, btn_manual, btn_skip):
            btn.label.set_color("white")
            btn.label.set_fontsize(8)

        def _use_auto(event):
            result.update({"use_auto": True, "low": None, "high": None})
            print("[ThresholdUI] Auto Multi-Otsu selected.")
            plt.close(fig)

        def _use_manual(event):
            low  = float(sl_low.val)
            high = max(float(sl_high.val), low + 0.001)
            result.update({"use_auto": False, "low": low, "high": high})
            print(f"[ThresholdUI] Manual thresholds — "
                  f"low={low:.4f}  high={high:.4f}")
            plt.close(fig)

        def _skip(event):
            result.update({"use_auto": True, "low": None, "high": None})
            plt.close(fig)

        btn_auto.on_clicked(_use_auto)
        btn_manual.on_clicked(_use_manual)
        btn_skip.on_clicked(_skip)

        plt.show(block=True)
        return result["low"], result["high"]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(cfg: Config = None) -> np.ndarray:
    """
    Execute the full Phase-1 pipeline and return the geometry_mask.

    Returns
    -------
    geometry_mask : np.ndarray, dtype=uint8, shape=(Z, Y, X)
        1 = solid grain  |  0 = fluid pore
    """
    if cfg is None:
        cfg = Config()

    print("=" * 60)
    print("  3D Digital Rock Pipeline  —  JAX-LaB Ready")
    print("=" * 60)

    # ── Step 1: Data Ingestion ───────────────────────────────────────────────
    print("\n[STEP 1] Data Ingestion")
    raw_volume = DataIngestion.load(cfg)

    # ── Step 1b: Interactive Crop (optional) ─────────────────────────────────
    if cfg.ENABLE_CROP_UI:
        print("\n[STEP 1b] Interactive Crop UI  "
              "(close window after confirming crop to continue)")
        raw_volume = CropUI.run(raw_volume)
    else:
        print("\n[STEP 1b] Crop UI disabled — using full volume")

    # ── Step 2: Preprocessing ───────────────────────────────────────────────
    print("\n[STEP 2] Image Preprocessing")
    denoised = Preprocessing.run(raw_volume, cfg)

    # ── Step 3: Segmentation ────────────────────────────────────────────────
    print("\n[STEP 3] Segmentation")

    # For beadpack mode — show interactive threshold selector before segmenting
    if cfg.SAMPLE_TYPE == "beadpack" and cfg.ENABLE_CROP_UI:
        print("[STEP 3a] Threshold Selector UI  "
              "(close window after confirming thresholds to continue)")
        t_low, t_high = ThresholdUI.run(denoised)
        if t_low is not None:
            cfg.BEAD_THRESH_LOW  = t_low
            cfg.BEAD_THRESH_HIGH = t_high

    binary_solid = Segmentation.run(denoised, cfg)

    # ── Step 4: Simulation-Ready Mask ───────────────────────────────────────
    print("\n[STEP 4] Simulation-Ready Mask")
    geometry_mask = SimulationMask.build(binary_solid)
    stats         = SimulationMask.compute_statistics(geometry_mask)
    SimulationMask.save(geometry_mask, cfg)

    # ── Visualise ────────────────────────────────────────────────────────────
    if cfg.SAVE_FIGURES:
        fig_path = cfg.OUTPUT_DIR / "pipeline_overview.png"
        Visualiser.plot_pipeline(
            raw      = raw_volume,
            denoised = denoised,
            binary   = binary_solid,
            mask     = geometry_mask,
            stats    = stats,
            save_path = fig_path,
        )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Porosity : {stats['porosity']*100:.2f} %")
    print(f"  Outputs  : {cfg.OUTPUT_DIR.resolve()}")
    print("=" * 60)

    return geometry_mask


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()

    # ── Point to your micro-CT TIFF ──────────────────────────────────────────
    cfg.INPUT_MODE    = "tiff_stack"
    cfg.TIFF_DIR      = Path("./tiff_stack")     # folder containing your .tif file(s)

    # ── Beadpack microfluidic settings ───────────────────────────────────────
    cfg.SAMPLE_TYPE   = "beadpack"   # 3-phase: air | plastic | glass bead
    cfg.ENABLE_CROP_UI = True        # Step 1b: crop inner channel (remove plastic border)
    #                                # Step 3a: interactive threshold selector opens automatically

    # ── Optional: override thresholds if you already know good values ────────
    # cfg.BEAD_THRESH_LOW  = 0.25    # air ↔ plastic  boundary (normalised 0–1)
    # cfg.BEAD_THRESH_HIGH = 0.62    # plastic ↔ bead boundary (normalised 0–1)

    # ── Denoising (median is faster for large volumes) ───────────────────────
    cfg.DENOISE_METHOD = "median"    # faster for 512-slice volumes
    # cfg.DENOISE_METHOD = "nlm"     # slower but sharper bead edges

    geometry_mask = run_pipeline(cfg)

    # geometry_mask is now ready for JAX-LaB / LBM simulation
    # 1 = glass bead (solid / bounce-back)
    # 0 = air pore space (fluid / LBM streaming node)
