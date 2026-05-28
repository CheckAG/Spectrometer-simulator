import streamlit as st
from PIL import Image

from config import (
    DEFAULT_SPECTRAL_RES, DEFAULT_SPAN_START, DEFAULT_SPAN_END,
    SPECTRAL_RES_MIN, SPECTRAL_RES_MAX,
    WAVELENGTH_MIN, WAVELENGTH_MAX,
    GRATING_TYPES, DEFAULT_GRATING_TYPE,
    GRATING_TYPE_TRANSMISSIVE, GRATING_TYPE_REFLECTIVE,
)

st.set_page_config(page_title="Optics sim", page_icon="🔭")
st.title(":blue[Welcome to the Optics Simulator]")
st.subheader("Guided way to design your spectrometer with ray tracing simulation",
             divider="violet")
st.caption("A easy way to design your first spectrometer")

with st.expander("**🎓 Click here to see how a spectrometer works:**"):
    st.image("img/optics_work.gif")

with st.sidebar:
    st.subheader("Please input the Spectral resolution and Span "
                 "according to your spectrometer design specification")

st.space("large")

if "page_0" not in st.session_state:
    st.session_state.page_0 = {
        "continue_btn_state": True,
        "spectral_resolution": DEFAULT_SPECTRAL_RES,
        "span_start": DEFAULT_SPAN_START,
        "span_end": DEFAULT_SPAN_END,
        "grating_type": DEFAULT_GRATING_TYPE,
    }

user_input = st.container()

with user_input:
    st.subheader("Select Grating Type")
    grating_type_options = GRATING_TYPES
    grating_type_index = grating_type_options.index(
        st.session_state.page_0["grating_type"])
    selected_grating_type = st.radio(
        label="_**Select the grating configuration**_",
        options=grating_type_options,
        index=grating_type_index,
        horizontal=True,
        help=(
            "**Transmissive**: light passes through the grating (4f inline layout). "
            "**Reflective**: light diffracts back from the grating surface "
            "(Czerny-Turner style layout)."
        ),
    )
    if selected_grating_type == GRATING_TYPE_REFLECTIVE:
        st.info(
            "Reflective grating selected. The simulation will use a "
            "Czerny-Turner style folded layout where F2 and the sensor "
            "are placed on the same side as the light source.",
            icon="ℹ️",
        )

    st.space("medium")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Enter the spectral resolution and required Span")
        st.number_input(
            label="_**Enter Spectral resolution (in nanometers)**_",
            min_value=SPECTRAL_RES_MIN,
            max_value=SPECTRAL_RES_MAX,
            step=1.0,
            key="spectral_resolution",
            value=st.session_state.page_0["spectral_resolution"],
            help="Refer to the image for clarity on Spectral resolution",
        )
        st.markdown("###### **Enter the start and end wavelength for required Span**")
        st.number_input(
            label="_**Enter starting wavelength (nm)**_",
            min_value=WAVELENGTH_MIN,
            max_value=WAVELENGTH_MAX,
            step=1.0,
            key="span_start",
            value=st.session_state.page_0["span_start"],
            help="Refer to the image for clarity on Span",
        )
        st.number_input(
            label="_**Enter end wavelength (nm)**_",
            min_value=WAVELENGTH_MIN,
            max_value=WAVELENGTH_MAX,
            step=1.0,
            key="span_end",
            value=st.session_state.page_0["span_end"],
            help="Refer to the image for clarity on Span",
        )
    with col2:
        st.image(Image.open("img/resolution_and_span_2.png"),
                 "Spectral resolution and Span example")

    span_start = st.session_state.span_start
    span_end = st.session_state.span_end

    if span_start >= span_end:
        st.error("End wavelength must be greater than start wavelength.", icon="🚨")
        st.session_state.page_0["continue_btn_state"] = True
    elif span_start * 2 < span_end:
        st.warning(
            "Diffraction orders might overlap for the span selected. "
            "Consider using order sorting filters. Do you wish to continue?",
            icon="⚠️",
        )
        st.image(Image.open("img/order_overlap.png"), "Diffraction orders overlap")
        st.session_state.page_0["continue_btn_state"] = False
    else:
        st.success("Ready — click Continue to proceed.", icon="✔️")
        st.session_state.page_0["continue_btn_state"] = False

    continue_btn = st.button(
        "Continue",
        help="Click to submit and navigate to next page",
        disabled=st.session_state.page_0["continue_btn_state"],
    )

    if continue_btn:
        st.session_state.page_0.update({
            "spectral_resolution": st.session_state.spectral_resolution,
            "span_start": st.session_state.span_start,
            "span_end": st.session_state.span_end,
            "grating_type": selected_grating_type,
        })
        st.switch_page("pages/1_Grating_specification.py")
