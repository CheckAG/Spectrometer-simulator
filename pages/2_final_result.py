import os
import streamlit as st
import numpy as np
from PIL import Image

from streamlit_extras.metric_cards import style_metric_cards
from grating import design_rendering
from config import (
    DEFAULT_PIXEL_SIZE, DEFAULT_F1, DEFAULT_F2,
    PIXEL_SIZE_MIN, PIXEL_SIZE_MAX,
    FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX,
    INCIDENT_ANGLE_MIN, INCIDENT_ANGLE_MAX,
    LENS_APERTURE, GRATING_APERTURE,
    SIMULATION_OUTPUT_PATH,
)
from optics import (
    calculate_blaze_wavelength,
    calculate_incident_angle,
    calculate_beta,
    calculate_optical_dispersion,
    calculate_f2_prescribed,
    calculate_sensor_width,
    calculate_slit_width,
    calculate_optical_resolution,
)

# ── Guard: require prior pages to be complete ────────────────────────────────
if "page_0" not in st.session_state or "span_start" not in st.session_state.page_0:
    st.warning("Please complete the first page before continuing.")
    st.switch_page("User-input.py")
if "page_1" not in st.session_state or "N" not in st.session_state.page_1:
    st.warning("Please complete the grating specification page before continuing.")
    st.switch_page("User-input.py")

# ── Restore persisted values from all prior pages ────────────────────────────
span_start = st.session_state.page_0["span_start"]
span_end = st.session_state.page_0["span_end"]
spectral_resolution = st.session_state.page_0["spectral_resolution"]
blaze_angle = st.session_state.page_1["blaze_angle"]
N = st.session_state.page_1["N"]
m = st.session_state.page_1["m"]
blaze_wavelength = calculate_blaze_wavelength(span_start, span_end)

# Write back so the rest of this page can read them from session_state if needed
st.session_state.update({
    "span_start": span_start,
    "span_end": span_end,
    "spectral_resolution": spectral_resolution,
    "blaze_angle": blaze_angle,
    "N": N,
    "m": m,
    "blaze_wavelength": blaze_wavelength,
})

# ── page_2 persistence dict (widget defaults) ────────────────────────────────
if "page_2" not in st.session_state:
    st.session_state.page_2 = {
        "pixel_size": DEFAULT_PIXEL_SIZE,
        "f1": DEFAULT_F1,
        "f2": DEFAULT_F2,
    }

# ── Pre-calculate incident angle ─────────────────────────────────────────────
incident_angle_calculated = calculate_incident_angle(
    blaze_wavelength, N, blaze_angle, m)

# ─────────────────────────────────────────────────────────────────────────────
st.title(":blue[Angle, Line array & Lens Specification:]")
st.space("xlarge")

with st.sidebar:
    st.subheader("Incident angle, Refracted angle and Blaze angles")
    st.subheader("Line array Detector specifications")
    st.subheader("Lens focal length specification")

user_input = st.container()

with user_input:

    # ── Incident angle ────────────────────────────────────────────────────────
    if np.isnan(incident_angle_calculated):
        st.error("Calculated incident angle exceeds 90°. Please set it manually.")
        st.slider(
            label="_**Please change the incident angle here (in degrees)**_",
            min_value=INCIDENT_ANGLE_MIN,
            max_value=INCIDENT_ANGLE_MAX,
            value=st.session_state.page_2.get("incident_angle", 0),
            key="incident_angle",
            step=1,
        )
    else:
        st.info(
            f"With the selected Spectral resolution, Span, and N value, "
            f"the calculated incident angle is _{incident_angle_calculated}°_. "
            f"Change below if needed.",
            icon="ℹ️",
        )
        st.slider(
            label="_**Please change the incident angle here (in degrees)**_",
            min_value=INCIDENT_ANGLE_MIN,
            max_value=INCIDENT_ANGLE_MAX,
            value=st.session_state.page_2.get(
                "incident_angle", int(incident_angle_calculated)),
            key="incident_angle",
            step=1,
        )

    incident_angle = st.session_state.incident_angle

    beta_min = calculate_beta(span_start, N, incident_angle, m)
    beta_max = calculate_beta(span_end, N, incident_angle, m)
    beta_middle = (beta_min + beta_max) / 2.0

    with st.expander("Dispersion span of the system:"):
        st.markdown(f"###### Beta minimum = {beta_min}°")
        st.markdown(f"###### Beta maximum = {beta_max}°")

    st.space("medium")

    # ── Pixel size ────────────────────────────────────────────────────────────
    st.number_input(
        label="_**Pixel size of the detector (µm)**_",
        min_value=PIXEL_SIZE_MIN,
        max_value=PIXEL_SIZE_MAX,
        value=st.session_state.page_2["pixel_size"],
        key="pixel_size",
        step=1.0,
    )
    pixel_size = st.session_state.pixel_size

    optical_dispersion = calculate_optical_dispersion(
        blaze_wavelength, N, incident_angle, m)
    f2_prescribed = calculate_f2_prescribed(
        pixel_size, optical_dispersion, spectral_resolution)

    st.info(f"Prescribed value of f2: {f2_prescribed} mm", icon="ℹ️")

    st.space("large")
    st.subheader("Change the value of F1 and F2 if you wish to")

    col1, col2 = st.columns(2)
    with col2:
        st.number_input(
            label="_**Set the value of f2 (mm)**_",
            min_value=FOCAL_LENGTH_MIN,
            max_value=FOCAL_LENGTH_MAX,
            key="f2",
            value=st.session_state.page_2["f2"],
        )
    with col1:
        st.number_input(
            label="_**Set the value of f1 (mm)**_",
            min_value=FOCAL_LENGTH_MIN,
            max_value=FOCAL_LENGTH_MAX,
            key="f1",
            value=st.session_state.page_2["f1"],
        )

    f1 = st.session_state.f1
    f2 = st.session_state.f2

    # Save widget values for next re-run
    st.session_state.page_2.update({
        "pixel_size": pixel_size,
        "f1": f1,
        "f2": f2,
        "incident_angle": incident_angle,
    })

    sensor_width = calculate_sensor_width(f2, beta_min, beta_max)
    slit_width = calculate_slit_width(f1, f2, pixel_size)
    optical_resolution = calculate_optical_resolution(f2, pixel_size, optical_dispersion)

    st.space("large")

    # ── Ray tracing simulation ────────────────────────────────────────────────
    st.header("Ray tracing simulation:")

    with st.spinner("Simulation running..."):
        try:
            design_rendering(
                f1=f1,
                f2=f2,
                sensor_width=sensor_width,
                start_wavelength=span_start,
                end_wavelength=span_end,
                incident_angle=incident_angle,
                difracted_angle=beta_middle,
                N=N,
                lens_aperture=LENS_APERTURE,
                grating_aperture=GRATING_APERTURE,
                output_path=SIMULATION_OUTPUT_PATH,
            )
            if os.path.exists(SIMULATION_OUTPUT_PATH):
                st.image(Image.open(SIMULATION_OUTPUT_PATH))
        except Exception as e:
            st.error(f"Simulation failed: {e}")

    st.space("large")

    # ── Final design values ───────────────────────────────────────────────────
    st.header("Final design values:")

    st.subheader("User input:")
    a1, a2, a3 = st.columns(3)
    a1.metric(label="Spectral Resolution (nm)", value=spectral_resolution)
    a2.metric(label="Minimum wavelength (nm)", value=span_start)
    a3.metric(label="Maximum wavelength (nm)", value=span_end)

    st.subheader("Resolution:")
    r1, _, _ = st.columns(3)
    r1.metric(label="Resolution achieved (nm)", value=optical_resolution)

    st.subheader("Grating specification:")
    b1, b2, _ = st.columns(3)
    b1.metric(label="Grooves per mm", value=N)
    b2.metric(label="Blaze angle", value=f"{blaze_angle}°")

    st.subheader("Optical specification:")
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Incident angle", value=f"{incident_angle}°")
    c2.metric(label="Minimum diffracted angle", value=f"{beta_min}°")
    c3.metric(label="Maximum diffracted angle", value=f"{beta_max}°")

    d1, d2, d3 = st.columns(3)
    d1.metric(label="F1 (mm)", value=np.round(f1, 2))
    d2.metric(label="F2 (mm)", value=f2)
    d3.metric(label="Slit width (mm)", value=slit_width)

    st.subheader("Detector specification:")
    e1, e2, _ = st.columns(3)
    e1.metric(label="Pixel size (µm)", value=pixel_size)
    e2.metric(label="Sensor width (mm)", value=sensor_width)

    style_metric_cards(
        background_color="#262730",
        border_color="#3D3D3D",
        border_left_color="#9AD8E1",
    )
