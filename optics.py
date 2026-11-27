"""
Spectrometer design physics.

Conventions (used everywhere in this module)
--------------------------------------------
Grating equation (Palmer & Loewen, *Diffraction Grating Handbook*):

    m·λ·G = sin α + sin β

α = incidence angle, β = diffraction angle, both measured from the grating
normal and positive when on the same side of it. The same equation holds for
reflective and transmissive gratings; only the physical layout differs.

Layout is fixed by the included angle 2K = α − β (0 = Littrow for reflective,
symmetric/minimum-deviation for transmissive). For a centre wavelength:

    sin φ = m·λ·G / (2·cos K),   α = φ + K,   β = φ − K

Units at the API: wavelength nm, lengths mm, pixel size µm, angles degrees,
groove density G lines/mm. Angles that cannot exist (|sin| > 1) come back NaN.

The ray tracer (raytracing_v.py) measures β with the opposite sign
(β_tracer = −β); tests/test_raytracing.py checks the two agree.
"""

from dataclasses import dataclass, field

import numpy as np

from config import GROOVE_OPTIONS, GRATING_APERTURE, LENS_APERTURE

NM_TO_MM = 1e-6
UM_TO_MM = 1e-3


def _asin_deg(x):
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(x))


def diffraction_angle(wl_nm, G, alpha_deg, m=1):
    """β (deg) for incidence α. Vectorised over wl_nm."""
    wl_nm = np.asarray(wl_nm, dtype=float)
    return _asin_deg(m * wl_nm * NM_TO_MM * G - np.sin(np.radians(alpha_deg)))


def grating_angles(wl_nm, G, included_deg, m=1):
    """(α, β) in deg that send wl_nm along a layout with included angle α − β."""
    K = np.radians(included_deg) / 2
    phi = _asin_deg(m * wl_nm * NM_TO_MM * G / (2 * np.cos(K)))
    return phi + np.degrees(K), phi - np.degrees(K)


def angular_dispersion(G, beta_deg, m=1):
    """dβ/dλ in rad/nm at fixed incidence."""
    return m * G * NM_TO_MM / np.cos(np.radians(beta_deg))


def default_blaze_wavelength(wl_min, wl_max):
    """
    First-order rule of thumb: efficiency stays above ~50 % of peak over
    0.67·λB … 1.8·λB. Pick λB at the geometric centre of the window that
    covers the span.
    """
    # ponytail: rule-of-thumb window, replace with a real efficiency model later
    return float(np.sqrt((wl_max / 1.8) * (wl_min / 0.67)))


def blaze_angle(wl_blaze_nm, G, included_deg, m=1, transmissive=False, n_glass=1.5):
    """Facet angle (deg) that puts peak efficiency at wl_blaze_nm."""
    alpha, beta = grating_angles(wl_blaze_nm, G, included_deg, m)
    if not transmissive:
        # Facet normal bisects incident and diffracted rays (specular off the facet).
        return float((alpha + beta) / 2)
    # Transmissive: the facet acts as a small prism that must bend the beam by
    # δ = α + β, i.e. n·sin θB = sin(θB + δ).
    # ponytail: assumes the ray inside the glass is ~normal to the flat face.
    delta = np.radians(alpha + beta)
    return float(np.degrees(np.arctan(np.sin(delta) / (n_glass - np.cos(delta)))))


def fold_angle(clearance, distance):
    """
    Off-axis angle (deg) of a Czerny–Turner mirror so the folded beam clears an
    element of half-size `clearance` placed `distance` away.
    """
    # ponytail: geometric clearance only; Shafer coma-cancelling angles need real mirrors
    return float(max(5.0, np.degrees(0.5 * np.arcsin(min(clearance / distance, 0.9)))))


@dataclass
class Spec:
    """What the user wants plus the hardware they picked."""
    wl_min: float                 # nm
    wl_max: float                 # nm
    resolution: float             # nm, target FWHM
    pixel_um: float               # detector pixel pitch
    n_pixels: int                 # detector pixel count
    f1: float                     # collimator focal length, mm
    f2: float                     # focusing focal length, mm
    included_deg: float = 0.0     # α − β at the centre wavelength
    transmissive: bool = True
    m: int = 1
    sampling: float = 3.0         # pixels per resolution element
    beam_mm: float = LENS_APERTURE
    grating_mm: float = GRATING_APERTURE
    n_glass: float = 1.5          # transmissive grating substrate index
    blaze_nm: float | None = None  # None → default_blaze_wavelength

    @property
    def wl_center(self):
        return (self.wl_min + self.wl_max) / 2

    @property
    def sensor_mm(self):
        return self.n_pixels * self.pixel_um * UM_TO_MM


@dataclass
class Design:
    G: int
    alpha: float              # deg
    beta_center: float        # deg
    beta_min: float           # deg, at wl_min
    beta_max: float           # deg, at wl_max
    blaze_nm: float
    blaze_angle: float        # deg
    dispersion: float         # rad/nm at centre
    rld: float                # reciprocal linear dispersion, nm/mm
    slit_um: float            # slit width whose image spans `sampling` pixels
    spectrum_mm: float        # length of the span on the detector
    fold1: float              # M1 off-axis angle, deg (0 for transmissive)
    fold2: float              # M2 off-axis angle = detector tilt to the chief ray, deg
    resolution_image: float   # nm, slit image / pixel sampling limit
    resolution_grating: float  # nm, λ / (m·N_illuminated)
    resolution: float         # nm, combined
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return np.isfinite(self.resolution)


def analyze(spec: Spec, G: int) -> Design:
    """Evaluate a spectrometer built from `spec` with a G lines/mm grating."""
    m, f1, f2 = spec.m, spec.f1, spec.f2
    warnings = []

    alpha, beta_c = grating_angles(spec.wl_center, G, spec.included_deg, m)
    beta_min, beta_max = diffraction_angle([spec.wl_min, spec.wl_max], G, alpha, m)

    blaze_nm = spec.blaze_nm or default_blaze_wavelength(spec.wl_min, spec.wl_max)
    theta_b = blaze_angle(blaze_nm, G, spec.included_deg, m, spec.transmissive, spec.n_glass)

    if spec.transmissive:
        fold1 = fold2 = 0.0
    else:
        fold1 = fold_angle((spec.grating_mm + spec.beam_mm) / 2, f1)
        fold2 = fold_angle((spec.grating_mm + spec.sensor_mm) / 2, f2)

    # Detector on the focal plane, which is tilted by fold2 to the chief ray in
    # a Czerny–Turner: x(λ) = f2·[tan(Δβ − fold2) + tan(fold2)]. The tilt
    # stretches everything at the detector by 1/cos²(fold2).
    t2 = np.radians(fold2)
    d_beta = np.radians([beta_min - beta_c, beta_max - beta_c])
    x_min, x_max = f2 * (np.tan(d_beta - t2) + np.tan(t2))
    spectrum_mm = float(abs(x_max - x_min))
    f2_eff = f2 / np.cos(t2) ** 2

    D = angular_dispersion(G, beta_c, m)
    rld = 1.0 / (f2_eff * D)

    # Slit imaged with magnification (f2/f1)·(cos α / cos β) (anamorphic term).
    image_mm = spec.sampling * spec.pixel_um * UM_TO_MM
    mag = (f2_eff / f1) * np.cos(np.radians(alpha)) / np.cos(np.radians(beta_c))
    slit_um = image_mm / mag / UM_TO_MM

    footprint = spec.beam_mm / np.cos(np.radians(alpha))
    if footprint > spec.grating_mm:
        warnings.append(f"Beam footprint on grating ({footprint:.1f} mm) exceeds grating "
                        f"size ({spec.grating_mm} mm); light is clipped.")
        footprint = spec.grating_mm

    res_image = rld * image_mm
    res_grating = spec.wl_center / (m * G * footprint)
    resolution = float(np.hypot(res_image, res_grating))

    if not np.all(np.isfinite([alpha, beta_min, beta_max])):
        warnings.append(f"{G} lines/mm cannot diffract the full span at this included "
                        f"angle (|sin β| > 1). Use fewer lines/mm or a smaller angle.")
        resolution = float("nan")
    if spec.n_pixels < spec.sampling * (spec.wl_max - spec.wl_min) / spec.resolution:
        need = int(np.ceil(spec.sampling * (spec.wl_max - spec.wl_min) / spec.resolution))
        warnings.append(f"Detector has {spec.n_pixels} pixels but the span at the target "
                        f"resolution needs ≥ {need} (at {spec.sampling:g} px per resolution "
                        f"element). No grating fixes this; widen the target or use a longer sensor.")
    if spectrum_mm > spec.sensor_mm:
        # Spectrum length scales linearly with f2 (x = f2·tan Δβ).
        fit = spec.sensor_mm / spectrum_mm
        warnings.append(f"Spectrum is {spectrum_mm:.1f} mm long but the sensor is only "
                        f"{spec.sensor_mm:.1f} mm; the ends of the span fall off. To fit: "
                        f"f2 ≤ {f2 * fit:.0f} mm, or a sensor ≥ {spectrum_mm:.1f} mm, or a "
                        f"span of ~{(spec.wl_max - spec.wl_min) * fit:.0f} nm.")
    if resolution > spec.resolution:
        warnings.append(f"Achieved resolution {resolution:.3g} nm is worse than the "
                        f"{spec.resolution:g} nm target.")
    if not spec.transmissive and np.isfinite(alpha):
        # Mirrors sit f1 and f2 from the grating, (α − β) apart as seen from it.
        gap = np.sqrt(f1**2 + f2**2 - 2 * f1 * f2 * np.cos(np.radians(spec.included_deg)))
        m2_mm = (spec.beam_mm * np.cos(np.radians(beta_c)) / np.cos(np.radians(alpha))
                 + f2 * abs(np.radians(beta_max - beta_min))) / np.cos(t2)
        if gap < (spec.beam_mm + m2_mm) / 2:
            warnings.append(f"M1 and M2 would overlap ({gap:.0f} mm apart, need "
                            f"≈{(spec.beam_mm + m2_mm) / 2:.0f} mm). Increase the included "
                            f"angle or the focal lengths.")
    if 2 * spec.wl_min < spec.wl_max:
        warnings.append(f"Second order of {spec.wl_min:g} nm lands on {2 * spec.wl_min:g} nm; "
                        f"an order-sorting (long-pass) filter is needed.")
    if spec.wl_min < 0.67 * blaze_nm or spec.wl_max > 1.8 * blaze_nm:
        warnings.append(f"Span extends beyond the efficient range of a {blaze_nm:.0f} nm "
                        f"blaze (≈{0.67 * blaze_nm:.0f}–{1.8 * blaze_nm:.0f} nm).")

    return Design(
        G=G, alpha=float(alpha), beta_center=float(beta_c),
        beta_min=float(beta_min), beta_max=float(beta_max),
        blaze_nm=blaze_nm, blaze_angle=theta_b,
        dispersion=float(D), rld=float(rld), slit_um=float(slit_um),
        spectrum_mm=spectrum_mm, fold1=fold1, fold2=fold2,
        resolution_image=float(res_image), resolution_grating=float(res_grating),
        resolution=resolution, warnings=warnings,
    )


def recommend_groove_density(spec: Spec, options=GROOVE_OPTIONS) -> Design:
    """
    Lowest stock groove density that meets the target resolution and fits the
    span on the sensor. Lower G = smaller angles, wider coverage, cheaper.
    Falls back to the best-resolving design that still fits, then to any valid one.
    """
    designs = [analyze(spec, G) for G in sorted(options)]
    valid = [d for d in designs if d.ok]
    fits = [d for d in valid if d.spectrum_mm <= spec.sensor_mm]
    meets = [d for d in fits if d.resolution <= spec.resolution]
    if meets:
        return meets[0]
    pool = fits or valid or designs
    return min(pool, key=lambda d: d.resolution if d.ok else np.inf)
