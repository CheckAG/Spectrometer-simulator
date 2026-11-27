import numpy as np
import streamlit as st
from PIL import Image
from streamlit_extras.metric_cards import style_metric_cards

from config import GROOVE_OPTIONS, DIFFRACTION_ORDER, GRATING_TYPE_TRANSMISSIVE
from grating import design_rendering
from optics import Spec, analyze, recommend_groove_density

if "page_1" not in st.session_state or "page_0" not in st.session_state:
    st.warning("Please complete the previous pages before continuing.")
    st.switch_page("User-input.py")

p0, p1 = st.session_state.page_0, st.session_state.page_1
spec = Spec(
    wl_min=p0["span_start"], wl_max=p0["span_end"], resolution=p0["spectral_resolution"],
    pixel_um=p1["pixel_um"], n_pixels=p1["n_pixels"], sampling=p1["sampling"],
    f1=p1["f1"], f2=p1["f2"], included_deg=p1["included_deg"],
    transmissive=p0["grating_type"] == GRATING_TYPE_TRANSMISSIVE, m=DIFFRACTION_ORDER,
)

st.title(":blue[Design result]")

recommended = recommend_groove_density(spec)
G = st.selectbox("Grating groove density (lines/mm)", GROOVE_OPTIONS,
                 index=GROOVE_OPTIONS.index(recommended.G),
                 help=f"Recommended: {recommended.G} — the lowest stock density that "
                      f"meets the target and fits the sensor.")
d = analyze(spec, G)

for w in d.warnings:
    st.warning(w, icon="⚠️")
if not d.warnings:
    st.success("Design meets the target.", icon="✔️")

st.header("Ray tracing simulation")
if d.ok:
    with st.spinner("Simulation running..."):
        try:
            st.image(design_rendering(spec, d))
        except Exception as e:
            st.error(f"Simulation failed: {e}")
else:
    st.error("No valid geometry for this grating — nothing to trace.")


def r(x, n=2):
    return "—" if not np.isfinite(x) else round(float(x), n)


st.header("Design values")
st.subheader("Resolution")
c = st.columns(4)
c[0].metric("Target (nm)", spec.resolution)
c[1].metric("Achieved (nm)", r(d.resolution, 3))
c[2].metric("Slit / pixel limit (nm)", r(d.resolution_image, 3))
c[3].metric("Grating limit (nm)", r(d.resolution_grating, 3))

st.subheader("Grating")
c = st.columns(4)
c[0].metric("Lines/mm", G)
c[1].metric("Blaze wavelength (nm)", r(d.blaze_nm, 0))
c[2].metric("Blaze angle", f"{r(d.blaze_angle)}°")
c[3].metric("Type", p0["grating_type"])

st.subheader("Angles")
c = st.columns(4)
c[0].metric("Incidence α", f"{r(d.alpha)}°")
c[1].metric("β centre", f"{r(d.beta_center)}°")
c[2].metric(f"β at {spec.wl_min:g} nm", f"{r(d.beta_min)}°")
c[3].metric(f"β at {spec.wl_max:g} nm", f"{r(d.beta_max)}°")

st.subheader("Optics & detector")
c = st.columns(4)
c[0].metric("Slit width (µm)", r(d.slit_um, 1))
c[1].metric("Dispersion (nm/mm)", r(d.rld))
c[2].metric("Spectrum length (mm)", r(d.spectrum_mm))
c[3].metric("Sensor length (mm)", r(spec.sensor_mm))

with st.expander("**🎓 Know more about blaze angle:**"):
    st.image(Image.open("img/Blazed_grating.png"), "Grating representation")
    st.markdown(
        "A blazed grating has sawtooth grooves tilted at the blaze angle θB. "
        "Efficiency peaks at the blaze wavelength and stays useful from roughly "
        "0.67·λB to 1.8·λB in first order, so λB is chosen to centre the span in that window."
    )

style_metric_cards(background_color="#262730", border_color="#3D3D3D",
                   border_left_color="#9AD8E1")
