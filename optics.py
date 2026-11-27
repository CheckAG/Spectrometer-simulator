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

from dataclasses import dataclass, field, replace

import numpy as np

from config import GROOVE_OPTIONS, GRATING_APERTURE, LENS_APERTURE, AUTO_ANGLE_MARGIN

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


class Issue(str):
    """
    A warning. Behaves as its message string; also carries:
    field — the Spec input it is about (so a UI can show it next to that input),
    fixes — {button label: {Spec field: new value}} one-click remedies.
    """
    def __new__(cls, text, field=None, fixes=None):
        obj = super().__new__(cls, text)
        obj.field, obj.fixes = field, fixes or {}
        return obj


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
    included_deg: float | None = None  # α − β at the centre; None → auto_included_angle
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
    included_deg: float       # α − β actually used
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
    mirrors_clear: bool = True  # Czerny–Turner: M1 and M2 don't overlap

    @property
    def ok(self):
        return np.isfinite(self.resolution)


def auto_included_angle(spec: Spec, G: int, margin=AUTO_ANGLE_MARGIN) -> float:
    """
    Default included angle (deg).
    Transmissive: 0 — symmetric (Bragg) incidence gives the best efficiency and
    no anamorphic stretch, and nothing can collide.
    Reflective: smallest whole degree at which M1 and M2 clear, plus `margin`
    (closer to Littrow = better blaze efficiency, less off-axis aberration).
    """
    if spec.transmissive:
        return 0.0
    for angle in range(0, 91):
        d = analyze(replace(spec, included_deg=float(angle)), G, fixes=False)
        if d.ok and d.mirrors_clear:
            return float(min(angle + margin, 90))
    return 90.0


def _walk_until_fits(spec, G, overrides, start, step, max_steps=400):
    """
    First value from `start` (moving by `step`) at which the spectrum fits the
    sensor once `overrides(value)` is applied. Needed because in a Czerny–Turner
    f2, sensor size and angles also move the fold angles, so one-shot linear
    estimates can miss. None if nothing within max_steps works.
    """
    value = start
    for _ in range(max_steps):
        s = replace(spec, **overrides(value))
        d = analyze(s, G, fixes=False)
        if d.ok and d.spectrum_mm <= s.sensor_mm:
            return value
        value += step
    return None


def analyze(spec: Spec, G: int, fixes: bool = True) -> Design:
    """
    Evaluate a spectrometer built from `spec` with a G lines/mm grating.
    fixes=False skips searching for one-click remedies (used for nested calls).
    """
    if spec.included_deg is None:
        spec = replace(spec, included_deg=auto_included_angle(spec, G))
    m, f1, f2 = spec.m, spec.f1, spec.f2
    mirrors_clear = True
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
        warnings.append(Issue(f"Beam footprint on grating ({footprint:.1f} mm) exceeds grating "
                              f"size ({spec.grating_mm} mm); light is clipped.", "included_deg"))
        footprint = spec.grating_mm

    res_image = rld * image_mm
    res_grating = spec.wl_center / (m * G * footprint)
    resolution = float(np.hypot(res_image, res_grating))

    if not np.all(np.isfinite([alpha, beta_min, beta_max])):
        warnings.append(Issue(f"{G} lines/mm cannot diffract the full span at this included "
                              f"angle (|sin β| > 1). Use fewer lines/mm or a smaller angle.",
                              "included_deg", {"Use computed angle": {"included_deg": None}}))
        resolution = float("nan")
    if spec.n_pixels < spec.sampling * (spec.wl_max - spec.wl_min) / spec.resolution:
        need = int(np.ceil(spec.sampling * (spec.wl_max - spec.wl_min) / spec.resolution))
        best = np.ceil(spec.sampling * (spec.wl_max - spec.wl_min) / spec.n_pixels * 100) / 100
        warnings.append(Issue(
            f"Detector has {spec.n_pixels} pixels but the span at the target resolution needs "
            f"≥ {need} (at {spec.sampling:g} px per resolution element). No grating fixes this; "
            f"widen the target or use a longer sensor.", "n_pixels",
            {f"Target {best:g} nm": {"resolution": float(best)}, f"Use {need} pixels": {"n_pixels": need}}))
    if spectrum_mm > spec.sensor_mm:
        # First guesses scale linearly (x = f2·tan Δβ); _walk_until_fits refines them.
        fit = spec.sensor_mm / spectrum_mm
        remedies = {}
        if fixes:
            c = spec.wl_center
            f2_fit = _walk_until_fits(spec, G, lambda v: {"f2": v}, float(np.floor(f2 * fit)), -1.0)
            half = _walk_until_fits(
                spec, G, lambda h: {"wl_min": float(np.ceil(c - h)), "wl_max": float(np.floor(c + h))},
                float(np.floor((spec.wl_max - spec.wl_min) * fit / 2)), -1.0)
            px = _walk_until_fits(spec, G, lambda n: {"n_pixels": n},
                                  int(np.ceil(spectrum_mm / (spec.pixel_um * UM_TO_MM))), 16)
            if f2_fit and f2_fit > 0:
                remedies[f"f2 = {f2_fit:g} mm"] = {"f2": f2_fit}
            if half and half > 0:
                lo, hi = float(np.ceil(c - half)), float(np.floor(c + half))
                remedies[f"Span {lo:g}–{hi:g} nm"] = {"wl_min": lo, "wl_max": hi}
            if px:
                remedies[f"Use {px} pixels"] = {"n_pixels": px}
        hint = (" or ".join(remedies) if remedies else
                f"f2 ≤ {f2 * fit:.0f} mm, or a sensor ≥ {spectrum_mm:.1f} mm, or a span of "
                f"~{(spec.wl_max - spec.wl_min) * fit:.0f} nm")
        warnings.append(Issue(
            f"Spectrum is {spectrum_mm:.1f} mm long but the sensor is only {spec.sensor_mm:.1f} mm; "
            f"the ends of the span fall off. To fit: {hint}.", "f2", remedies))
    if resolution > spec.resolution:
        relaxed = float(np.ceil(resolution * 100) / 100)
        warnings.append(Issue(f"Achieved resolution {resolution:.3g} nm is worse than the "
                              f"{spec.resolution:g} nm target.", "resolution",
                              {f"Target {relaxed:g} nm": {"resolution": relaxed}}))
    if not spec.transmissive and np.isfinite(alpha):
        # Mirrors sit f1 and f2 from the grating, (α − β) apart as seen from it.
        gap = np.sqrt(f1**2 + f2**2 - 2 * f1 * f2 * np.cos(np.radians(spec.included_deg)))
        m2_mm = (spec.beam_mm * np.cos(np.radians(beta_c)) / np.cos(np.radians(alpha))
                 + f2 * abs(np.radians(beta_max - beta_min))) / np.cos(t2)
        mirrors_clear = bool(gap >= (spec.beam_mm + m2_mm) / 2)
        if not mirrors_clear:
            warnings.append(Issue(f"M1 and M2 would overlap ({gap:.0f} mm apart, need "
                                  f"≈{(spec.beam_mm + m2_mm) / 2:.0f} mm). Increase the included "
                                  f"angle or the focal lengths.", "included_deg",
                                  {"Use computed angle": {"included_deg": None}}))
    if 2 * spec.wl_min < spec.wl_max:
        warnings.append(Issue(f"Second order of {spec.wl_min:g} nm lands on {2 * spec.wl_min:g} nm; "
                              f"an order-sorting (long-pass) filter is needed.", "wl_max",
                              {f"Limit span to {spec.wl_min:g}–{2 * spec.wl_min:g} nm":
                               {"wl_max": float(2 * spec.wl_min)}}))
    if spec.wl_min < 0.67 * blaze_nm or spec.wl_max > 1.8 * blaze_nm:
        warnings.append(Issue(f"Span extends beyond the efficient range of a {blaze_nm:.0f} nm "
                              f"blaze (≈{0.67 * blaze_nm:.0f}–{1.8 * blaze_nm:.0f} nm).", "wl_max"))

    return Design(
        G=G, included_deg=spec.included_deg, alpha=float(alpha), beta_center=float(beta_c),
        beta_min=float(beta_min), beta_max=float(beta_max),
        blaze_nm=blaze_nm, blaze_angle=theta_b,
        dispersion=float(D), rld=float(rld), slit_um=float(slit_um),
        spectrum_mm=spectrum_mm, fold1=fold1, fold2=fold2,
        resolution_image=float(res_image), resolution_grating=float(res_grating),
        resolution=resolution, warnings=warnings, mirrors_clear=mirrors_clear,
    )


def recommend_groove_density(spec: Spec, options=GROOVE_OPTIONS) -> Design:
    """
    Lowest stock groove density that meets the target resolution and fits the
    span on the sensor. Lower G = smaller angles, wider coverage, cheaper.
    Falls back to the best-resolving design that still fits, then to the valid
    design whose spectrum comes closest to fitting the sensor.
    """
    designs = [analyze(spec, G) for G in sorted(options)]
    valid = [d for d in designs if d.ok]
    fits = [d for d in valid if d.spectrum_mm <= spec.sensor_mm]
    meets = [d for d in fits if d.resolution <= spec.resolution]
    if meets:
        return meets[0]
    if fits:
        return min(fits, key=lambda d: d.resolution)
    return min(valid, key=lambda d: d.spectrum_mm) if valid else designs[0]
