import streamlit as st
import numpy as np
from PIL import Image

from config import (
    GROOVE_OPTIONS, DIFFRACTION_ORDER,
    DEFAULT_BLAZE_ANGLE, BLAZE_ANGLE_MIN, BLAZE_ANGLE_MAX,
)
from optics import (
    closest_value, calculate_blaze_wavelength,
    calculate_optimal_N, calculate_blaze_angle,
)

if "page_0" not in st.session_state or "span_start" not in st.session_state.page_0:
    st.warning("Please complete the first page before continuing.")
    st.switch_page("User-input.py")

# Restore page 0 values
st.session_state.span_start = st.session_state.page_0["span_start"]
st.session_state.span_end = st.session_state.page_0["span_end"]
st.session_state.spectral_resolution = st.session_state.page_0["spectral_resolution"]

st.title(":blue[Grating Specification:]")
st.space("xlarge")

with st.sidebar:
    st.subheader("Please input the Grating Specification...")

if "page_1" not in st.session_state:
    st.session_state.page_1 = {
        "blaze_angle": DEFAULT_BLAZE_ANGLE,
        "N": closest_value(GROOVE_OPTIONS,
                           calculate_optimal_N(
                               calculate_blaze_wavelength(
                                   st.session_state.span_start,
                                   st.session_state.span_end),
                               st.session_state.spectral_resolution)),
        "m": DIFFRACTION_ORDER,
    }

# Derived values — recalculated on every run from persisted inputs
blaze_wavelength = calculate_blaze_wavelength(
    st.session_state.span_start, st.session_state.span_end)
span = st.session_state.span_end - st.session_state.span_start
N_recommended = calculate_optimal_N(blaze_wavelength, st.session_state.spectral_resolution)
blaze_angle_calculated = calculate_blaze_angle(blaze_wavelength, N_recommended)

user_input = st.container()

with user_input:
    st.info(
        f"The entered Spectral Resolution is {st.session_state.spectral_resolution} nm "
        f"with span of {span} nm",
        icon="ℹ️",
    )
    st.space("medium")

    st.warning(
        f"According to your input, the calculated number of grooves per mm is "
        f"_{N_recommended}_. Standard options are {GROOVE_OPTIONS}. "
        f"We recommend _{N_recommended}_ — change below if needed.",
        icon="⚠️",
    )
    st.selectbox(
        label="_**Please change the N value if you wish to (Grooves/mm)**_",
        options=GROOVE_OPTIONS,
        key="N_input",
        index=GROOVE_OPTIONS.index(st.session_state.page_1["N"]),
        help="Number of grooves per mm determines the wavelength splitting magnitude.",
    )

    blaze_angle_for_selected_N = calculate_blaze_angle(
        blaze_wavelength, st.session_state.N_input)

    st.info(
        f"For the selected N of {st.session_state.N_input} grooves/mm, the calculated "
        f"blaze angle at blaze wavelength _{blaze_wavelength}_ nm is "
        f"_{blaze_angle_for_selected_N}°_. "
        f"This is for Littrow configuration.",
        icon="ℹ️",
    )

    st.number_input(
        label="_**Please change the blaze angle if required**_",
        min_value=BLAZE_ANGLE_MIN,
        max_value=BLAZE_ANGLE_MAX,
        key="blaze_angle",
        value=st.session_state.page_1["blaze_angle"],
        help="Blaze angle optimises grating efficiency for the design wavelength.",
    )

    with st.expander("**🎓 Know more about blaze angle:**"):
        st.image(Image.open("img/Blazed_grating.png"), "Grating representation")
        st.text("""
        Like every optical grating, a blazed grating has a constant line spacing d,
        determining the magnitude of the wavelength splitting caused by the grating.
        The grating lines possess a triangular, sawtooth-shaped cross section, forming
        a step structure. The steps are tilted at the so-called blaze angle theta_B
        with respect to the grating surface. Accordingly, the angle between step normal
        and grating normal is theta_B.

        The blaze angle is optimised to maximise efficiency for the wavelength of the
        used light. In the Littrow configuration the incident and diffracted beams
        travel in the same direction.
        """)

    if st.button("Continue", help="Click to submit and navigate to next page"):
        st.session_state.page_1.update({
            "blaze_angle": st.session_state.blaze_angle,
            "N": st.session_state.N_input,
            "m": DIFFRACTION_ORDER,
        })
        # Keep plain session state in sync for pages that read it directly
        st.session_state.N = st.session_state.N_input
        st.switch_page("pages/2_final_result.py")
