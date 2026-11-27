"""Spectrometer designer: single-page Streamlit workbench."""

import json
from dataclasses import asdict, replace

import numpy as np
import streamlit as st

from config import (
    GROOVE_OPTIONS, GRATING_TYPES, GRATING_TYPE_TRANSMISSIVE, DIFFRACTION_ORDER,
    DEFAULT_SPAN_START, DEFAULT_SPAN_END, DEFAULT_SPECTRAL_RES, DEFAULT_GRATING_TYPE,
    DEFAULT_PIXEL_SIZE, DEFAULT_N_PIXELS, DEFAULT_SAMPLING, DEFAULT_F1, DEFAULT_F2,
    SPECTRAL_RES_MIN, SPECTRAL_RES_MAX, WAVELENGTH_MIN, WAVELENGTH_MAX,
    PIXEL_SIZE_MIN, PIXEL_SIZE_MAX, N_PIXELS_MIN, N_PIXELS_MAX,
    FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX, INCLUDED_ANGLE_MIN, INCLUDED_ANGLE_MAX,
)
from optics import (Issue, Spec, analyze, recommend_groove_density, detector_signal, line_dip,
                    HG_AR_LINES, blaze_efficiency, default_blaze_wavelength)
from plot import ray_figure, detector_figure, efficiency_figure
from grating import aberration_blur

st.set_page_config(page_title="Spectrometer Designer", page_icon="🔭", layout="wide")

# ── State ────────────────────────────────────────────────────────────────────
# Every input lives in st.session_state under the key below and is mirrored to
# the URL query string, so a design can be shared by copying the link.
# "angle" and "G" are None while they follow the computed/recommended value.
DEFAULTS = {
    "grating_type": DEFAULT_GRATING_TYPE,
    "wl_min": DEFAULT_SPAN_START, "wl_max": DEFAULT_SPAN_END,
    "resolution": DEFAULT_SPECTRAL_RES,
    "pixel_um": DEFAULT_PIXEL_SIZE, "n_pixels": DEFAULT_N_PIXELS, "sampling": DEFAULT_SAMPLING,
    "f1": DEFAULT_F1, "f2": DEFAULT_F2,
    "angle": None, "G": None, "blaze": None, "exact": False,
}
LIMITS = {
    "wl_min": (WAVELENGTH_MIN, WAVELENGTH_MAX), "wl_max": (WAVELENGTH_MIN, WAVELENGTH_MAX),
    "resolution": (SPECTRAL_RES_MIN, SPECTRAL_RES_MAX),
    "pixel_um": (PIXEL_SIZE_MIN, PIXEL_SIZE_MAX), "n_pixels": (N_PIXELS_MIN, N_PIXELS_MAX),
    "sampling": (1.0, 10.0), "f1": (FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX),
    "f2": (FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX),
    "angle": (INCLUDED_ANGLE_MIN, INCLUDED_ANGLE_MAX),
    "blaze": (WAVELENGTH_MIN, WAVELENGTH_MAX),
}
ss = st.session_state


def clamp(key, value):
    lo, hi = LIMITS[key]
    value = min(max(value, lo), hi)
    return int(value) if key == "n_pixels" else float(value)


def parse(key, raw):
    """Query-string value → typed, clamped value; None if unusable."""
    try:
        if key == "exact":
            return raw == "True"
        if key == "grating_type":
            return raw if raw in GRATING_TYPES else None
        if key == "G":
            return int(raw) if int(raw) in GROOVE_OPTIONS else None
        return clamp(key, float(raw))
    except (TypeError, ValueError):
        return None


if "loaded" not in ss:
    ss.loaded = True
    for key, default in DEFAULTS.items():
        parsed = parse(key, st.query_params[key]) if key in st.query_params else None
        ss[key] = default if parsed is None else parsed
    if ss.wl_min >= ss.wl_max:
        ss.wl_min, ss.wl_max = DEFAULT_SPAN_START, DEFAULT_SPAN_END


def apply_fix(fix):
    """Button callback: write a fix's Spec overrides into the inputs."""
    for field, value in fix.items():
        key = {"included_deg": "angle", "blaze_nm": "blaze"}.get(field, field)
        ss[key] = None if value is None else clamp(key, value)


def follow_computed(key):
    ss[key] = None


# ── Inputs (sidebar) ─────────────────────────────────────────────────────────
# Each section gets a slot where warnings about its inputs appear.
slots = {}
with st.sidebar:
    st.title("🔭 Spectrometer Designer")
    st.caption("Change any input; everything updates live.")

    st.subheader("Spectrum")
    c1, c2 = st.columns(2)
    c1.number_input("From (nm)", *LIMITS["wl_min"], step=10.0, key="wl_min")
    c2.number_input("To (nm)", *LIMITS["wl_max"], step=10.0, key="wl_max")
    st.number_input("Target resolution (nm)", *LIMITS["resolution"], step=0.1, key="resolution",
                    help="Smallest wavelength difference to tell apart (FWHM).")
    slots["spectrum"] = st.container()

    st.subheader("Detector")
    c1, c2 = st.columns(2)
    c1.number_input("Pixel pitch (µm)", *LIMITS["pixel_um"], step=0.5, key="pixel_um")
    c2.number_input("Pixels", *LIMITS["n_pixels"], step=64, key="n_pixels")
    st.slider("Pixels per resolution element", *LIMITS["sampling"], step=0.5, key="sampling",
              help="2 = Nyquist minimum, 3 = common practice.")
    slots["detector"] = st.container()

    st.subheader("Optics")
    st.radio("Grating", GRATING_TYPES, horizontal=True, key="grating_type",
             on_change=follow_computed, args=("angle",),
             help="Transmissive: 4f layout with lenses. Reflective: Czerny–Turner with "
                  "concave mirrors.")
    c1, c2 = st.columns(2)
    c1.number_input("f1, collimator (mm)", *LIMITS["f1"], step=5.0, key="f1")
    c2.number_input("f2, focusing (mm)", *LIMITS["f2"], step=5.0, key="f2",
                    help="Longer f2 spreads the spectrum more: finer resolution, longer spectrum.")
    st.toggle("Exact mirrors (aberrations)", key="exact",
              disabled=ss.grating_type == GRATING_TYPE_TRANSMISSIVE,
              help="Czerny–Turner only. Trace real spherical mirrors instead of ideal ones: "
                   "spherical aberration, coma and the off-axis focus shift blur the spectrum. "
                   "The sensor is placed at the traced best focus.")
    slots["optics"] = st.container()

    angle_slot = st.container()   # filled once the computed angle is known
    grating_slot = st.container()

# ── Compute ──────────────────────────────────────────────────────────────────
if ss.wl_min >= ss.wl_max:
    st.error("The spectrum must end at a longer wavelength than it starts.", icon="🚨")
    st.stop()

spec = Spec(wl_min=ss.wl_min, wl_max=ss.wl_max, resolution=ss.resolution,
            pixel_um=ss.pixel_um, n_pixels=int(ss.n_pixels), sampling=ss.sampling,
            f1=ss.f1, f2=ss.f2, included_deg=ss.angle, blaze_nm=ss.blaze,
            transmissive=ss.grating_type == GRATING_TYPE_TRANSMISSIVE, m=DIFFRACTION_ORDER)
recommended = recommend_groove_density(spec)
G = ss.G or recommended.G
design = analyze(spec, G)
computed_angle = analyze(replace(spec, included_deg=None), G).included_deg

# Exact mirrors: traced blur adds in quadrature to the analytic resolution.
exact = ss.exact and not spec.transmissive and design.ok
resolution = design.resolution
if exact:
    blur_wl, blur_nm = aberration_blur(spec, design)
    resolution = float(np.hypot(design.resolution, np.nanmax(blur_nm)))
    if resolution > spec.resolution:
        design.warnings.append(Issue(
            f"Mirror aberrations add up to {np.nanmax(blur_nm):.2f} nm of blur, so resolution "
            f"with exact mirrors is {resolution:.3f} nm (target {spec.resolution:g} nm). Fast, "
            f"strongly folded mirrors cause it: longer f1/f2 or a smaller included angle "
            f"reduce it.", "f2"))

with angle_slot:
    # The slider shows the computed angle until the user moves it.
    ss.angle_slider = ss.angle if ss.angle is not None else computed_angle
    st.slider("Included angle α − β (°)", *LIMITS["angle"], step=1.0, key="angle_slider",
              on_change=lambda: ss.update(angle=ss.angle_slider),
              help="Angle between the beam arriving at the grating and the diffracted beam. "
                   "Computed: 0° transmissive (symmetric, best efficiency); reflective = "
                   "smallest angle at which M1 and M2 clear, + 5°.")
    if ss.angle is None:
        st.caption(f"Computed for {G} lines/mm. Drag to override.")
    else:
        c1, c2 = st.columns([3, 2])
        c1.caption(f"Manual · computed: {computed_angle:g}°")
        c2.button("Use computed", key="angle_reset", on_click=follow_computed, args=("angle",))

with grating_slot:
    ss.G_select = G
    st.selectbox("Grooves per mm", GROOVE_OPTIONS, key="G_select",
                 format_func=lambda g: f"{g}  ★ recommended" if g == recommended.G else str(g),
                 on_change=lambda: ss.update(G=ss.G_select),
                 help="Recommended: the lowest stock density that meets the target and fits "
                      "the sensor.")
    if ss.G is not None and ss.G != recommended.G:
        st.button(f"Use recommended ({recommended.G})", key="G_reset",
                  on_click=follow_computed, args=("G",))
    computed_blaze = default_blaze_wavelength(spec.wl_min, spec.wl_max)
    ss.blaze_input = ss.blaze if ss.blaze is not None else round(computed_blaze)
    st.number_input("Blaze wavelength (nm)", *LIMITS["blaze"], step=10.0, key="blaze_input",
                    on_change=lambda: ss.update(blaze=ss.blaze_input),
                    help="Wavelength of peak grating efficiency. Computed to centre the span "
                         "in the 0.67–1.8 × blaze window; set it to the blaze of a grating "
                         "you own or plan to buy.")
    if ss.blaze is not None:
        c1, c2 = st.columns([3, 2])
        c1.caption(f"Manual · computed: {computed_blaze:.0f} nm")
        c2.button("Use computed", key="blaze_reset", on_click=follow_computed, args=("blaze",))

# Mirror inputs to the URL so the page link reproduces this design.
st.query_params.from_dict({k: ss[k] for k in DEFAULTS if ss[k] is not None})

# ── Warnings next to their inputs ────────────────────────────────────────────
SECTION = {"wl_min": "spectrum", "wl_max": "spectrum", "resolution": "spectrum",
           "n_pixels": "detector", "f2": "optics", "included_deg": "optics", "blaze_nm": "optics"}
for issue in design.warnings:
    with slots[SECTION.get(issue.field, "optics")]:
        st.caption(f"⚠️ {issue.split('. ')[0].split('; ')[0]}")

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("Spectrometer Designer")
kind = "4f transmissive" if spec.transmissive else "Czerny–Turner"
st.caption(f"{kind} · {spec.wl_min:g}–{spec.wl_max:g} nm · {G} lines/mm · "
           f"{spec.n_pixels} × {spec.pixel_um:g} µm sensor")


def fmt(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


k = st.columns(5)
k[0].metric("Resolution" + (" (exact mirrors)" if exact else ""), f"{fmt(resolution, 3)} nm",
            delta=f"{resolution - spec.resolution:+.3f} nm vs target" if design.ok else None,
            delta_color="inverse", border=True)
k[1].metric("Sensor used", f"{fmt(design.spectrum_mm / spec.sensor_mm * 100, 0)} %",
            delta=f"{fmt(design.spectrum_mm, 1)} of {spec.sensor_mm:.1f} mm", delta_color="off",
            border=True)
k[2].metric("Grating", f"{G} l/mm",
            delta=f"blaze {fmt(design.blaze_angle, 1)}° @ {design.blaze_nm:.0f} nm",
            delta_color="off", border=True)
k[3].metric("Slit width", f"{fmt(design.slit_um, 1)} µm", border=True)
k[4].metric("Included angle", f"{fmt(design.included_deg, 0)}°",
            delta=f"α {fmt(design.alpha, 1)}°", delta_color="off", border=True)

if design.warnings:
    with st.container(border=True):
        for i, issue in enumerate(design.warnings):
            st.warning(issue, icon="⚠️")
            if issue.fixes:
                cols = st.columns(len(issue.fixes) + 1)
                for j, (label, fix) in enumerate(issue.fixes.items()):
                    cols[j].button(f"Fix: {label}", key=f"fix_{i}_{j}", on_click=apply_fix,
                                   args=(fix,), width="stretch")
else:
    st.success("Design meets the target and fits the sensor.", icon="✅")

trace_tab, detector_tab, values_tab, learn_tab, export_tab = st.tabs(
    ["Ray trace", "Detector", "Design values", "How it works", "Share & export"])

with trace_tab:
    show_stray = st.toggle("Show stray orders (0 and 2)", value=True)
    if design.ok:
        st.plotly_chart(ray_figure(spec, design, extra_orders=(0, 2) if show_stray else (),
                                   exact=exact),
                        width="stretch", config={"scrollZoom": True})
        st.caption("Scroll to zoom, drag to pan, double-click to reset. Hover a ray for its "
                   "wavelength and order; click a legend entry to hide an order.")
    else:
        st.error("No valid geometry for this grating — nothing to trace.")

with detector_tab:
    source = st.segmented_control("Light source", ["Hg–Ar lamp", "Custom lines"],
                                  default="Hg–Ar lamp", key="source")
    if source == "Custom lines":
        text = st.text_input("Emission lines (nm, comma-separated)", "589.0, 589.6",
                             help="Try the sodium D doublet (589.0, 589.6) or any pair "
                                  "closer than your resolution.")
        try:
            lines = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Enter numbers separated by commas.")
            lines = []
    else:
        lines = [l for l in HG_AR_LINES if spec.wl_min <= l <= spec.wl_max]
    if not design.ok:
        st.error("No valid geometry for this grating — nothing to simulate.")
    elif lines:
        # Lamp lines taken as equally bright; the grating's efficiency weights them.
        wl_pix, signal = detector_signal(spec, design, lines,
                                         blaze_efficiency(lines, design.blaze_nm, spec.m),
                                         np.interp(lines, blur_wl, blur_nm) if exact else 0.0)
        st.plotly_chart(detector_figure(wl_pix, signal, lines,
                                        (spec.wl_min, spec.wl_max)), width="stretch")
        st.caption(f"{spec.n_pixels} pixels of {spec.pixel_um:g} µm. Each line is blurred to the "
                   f"design resolution ({resolution:.3f} nm FWHM) and scaled by the grating "
                   f"efficiency below; dotted lines mark the true wavelengths. Drag to zoom "
                   f"in on a pair, double-click to reset.")
        close = sorted(lines)
        pairs = [(a, b) for a, b in zip(close, close[1:]) if b - a < 3 * design.resolution]
        for a, b in pairs:
            dip = line_dip(wl_pix, signal, a, b)
            if np.isnan(dip):
                st.info(f"{a:g} / {b:g} nm: not both on the sensor.")
            elif dip <= 0.81:
                st.success(f"{a:g} / {b:g} nm ({b - a:.2f} nm apart): **resolved**, signal dips "
                           f"to {dip:.0%} between them.", icon="✅")
            else:
                st.warning(f"{a:g} / {b:g} nm ({b - a:.2f} nm apart): **not resolved**"
                           + (f", dip only to {dip:.0%}" if dip < 1 else ", seen as one peak")
                           + " (Rayleigh needs ≤ 81 %).", icon="⚠️")
    else:
        st.info("No lines inside the span.")

    st.markdown("#### Grating efficiency")
    st.plotly_chart(efficiency_figure(spec.wl_min, spec.wl_max, design.blaze_nm, spec.m),
                    width="stretch")
    eta = blaze_efficiency([spec.wl_min, spec.wl_max], design.blaze_nm, spec.m)
    st.caption(f"Blaze {design.blaze_nm:.0f} nm ({design.blaze_angle:.1f}° facets). Relative "
               f"efficiency {eta[0]:.0%} at {spec.wl_min:g} nm and {eta[1]:.0%} at "
               f"{spec.wl_max:g} nm. Order 2 peaks at half the blaze wavelength, so short "
               f"wavelengths in order 2 are what an order-sorting filter has to block. "
               f"Scalar model: real gratings differ with polarisation.")

with values_tab:
    rows = [
        ("Resolution", "Achieved", fmt(resolution, 3), "nm"),
        ("Resolution", "Slit / pixel limit", fmt(design.resolution_image, 3), "nm"),
        ("Resolution", "Grating limit", fmt(design.resolution_grating, 3), "nm"),
        ("Grating", "Grooves", f"{G}", "lines/mm"),
        ("Grating", "Blaze wavelength", fmt(design.blaze_nm, 0), "nm"),
        ("Grating", "Blaze angle", fmt(design.blaze_angle, 2), "°"),
        ("Angles", "Included α − β", fmt(design.included_deg, 1), "°"),
        ("Angles", "Incidence α", fmt(design.alpha, 2), "°"),
        ("Angles", f"β at {spec.wl_min:g} nm", fmt(design.beta_min, 2), "°"),
        ("Angles", f"β at {spec.wl_max:g} nm", fmt(design.beta_max, 2), "°"),
        ("Optics", "Slit width", fmt(design.slit_um, 1), "µm"),
        ("Optics", "Reciprocal linear dispersion", fmt(design.rld, 2), "nm/mm"),
        ("Optics", "Spectrum length", fmt(design.spectrum_mm, 2), "mm"),
        ("Optics", "Sensor length", fmt(spec.sensor_mm, 2), "mm"),
    ]
    if not spec.transmissive:
        rows += [("Optics", "M1 off-axis angle", fmt(design.fold1, 1), "°"),
                 ("Optics", "M2 off-axis angle (detector tilt)", fmt(design.fold2, 1), "°")]
    if exact:
        rows.insert(3, ("Resolution", "Mirror aberration blur",
                        f"{np.nanmin(blur_nm):.3f} – {np.nanmax(blur_nm):.3f}", "nm"))
    st.dataframe([dict(zip(["Group", "Quantity", "Value", "Unit"], r)) for r in rows],
                 hide_index=True, width="stretch")

with learn_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### How a spectrometer works")
        st.image("img/optics_work.gif")
        st.markdown(
            "Light enters through a narrow **slit**. The first lens or mirror (**f1**) makes "
            "the beam parallel, the **grating** sends each wavelength off at a slightly "
            "different angle, and the second lens or mirror (**f2**) focuses each angle to a "
            "different spot on the **sensor**.")
        st.markdown("#### Resolution and span")
        st.image("img/resolution_and_span_2.png")
        st.markdown(
            "Resolution is set by how wide the slit's image is on the sensor (aimed at a few "
            "pixels) divided by how far apart neighbouring wavelengths land (the dispersion). "
            "More lines per mm or a longer f2 spreads wavelengths further: finer resolution, "
            "but a longer spectrum that has to fit on the sensor.")
    with c2:
        st.markdown("#### Blazed gratings")
        st.image("img/Blazed_grating.png")
        st.markdown(
            "A blazed grating's sawtooth grooves are tilted at the blaze angle θB. Efficiency "
            "peaks at the blaze wavelength and stays useful from about 0.67·λB to 1.8·λB "
            "in first order, so the blaze wavelength is chosen to centre the span in that window.")
        st.markdown("#### Order overlap")
        st.image("img/order_overlap.png")
        st.markdown(
            "Light at λ in second order leaves the grating at the same angle as 2λ in first "
            "order. If the span reaches past twice its start, both land on the same pixels; "
            "a long-pass (order-sorting) filter blocks the short wavelengths there.")

with export_tab:
    st.markdown("**Share:** the page address already contains this design — copy it from "
                "the browser's address bar.")
    payload = {"spec": asdict(spec),
               "design": {k: v for k, v in asdict(design).items() if k != "warnings"},
               "warnings": [str(w) for w in design.warnings]}
    st.download_button("Download design (JSON)", json.dumps(payload, indent=2, default=float),
                       file_name="spectrometer-design.json", mime="application/json")
    st.json(payload, expanded=False)
