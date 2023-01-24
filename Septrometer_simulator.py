import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
# from numericalunits import nm, mm, cm
from streamlit.components.v1 import html
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header


st.set_page_config(page_title="Optics sim", page_icon="🔭")
st.session_state.update(st.session_state)
st.title(":blue[Welcome to the Optics Simulator]")

colored_header(
    label="Guided way to design your spectrometer with ray tracing simulation",
    description="A easy way to design your first spectrometer",
    color_name="violet-70",
)

with st.sidebar:
    st.subheader("Please input the Spectral resolution and Span according to your spectrometer design specification")

add_vertical_space(2)

user_input = st.container()

if 'page_0' not in st.session_state:
    st.session_state.page_0 = {
        'spec_res_input': 1.0,
        'span_start_input': 400.0,
        'span_end_input': 400.0,
        'continue_btn_state' : True
    }

with user_input:
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Enter the spectral resolution and required Span")
        spec_res_input = st.number_input(label="_**Enter Spectral resolution (in nanometers)**_",
                        min_value=0.1,
                        max_value=100.0,
                        step=1.0,
                        value=st.session_state.page_0['spec_res_input'],
                        key='spectral_resolution',
                        help="to get clarity on Spectral resolution refer to the image next to the input field by enlarging it")
        st.markdown("###### **Enter the start and end wavelength for required Span**")
        span_start_input = st.number_input(label="_**Enter starting wavelength (nm)**_",
                        min_value=100.0,
                        max_value=2000.0,
                        step=1.0,
                        value=st.session_state.page_0['span_start_input'],
                        key='span_start',
                        help="to get clarity on Span refer to the image next to the input field by enlarging it")
        span_end_input = st.number_input(label="_**Enter end wavelength (nm)**_",
                        min_value=100.0,
                        max_value=2000.0,
                        step=1.0,
                        value=st.session_state.page_0['span_end_input'],
                        key='span_end',
                        help="to get clarity on Span refer to the image next to the input field by enlarging it")
    with col2:
        image = Image.open('img/resolution_and_span_2.png')
        st.image(image, "Spectral resolution and Span example")
    

    if span_start_input >= span_end_input:
        st.error("Please make sure that end wavelength is grater than start wavelength",icon="🚨")
        st.session_state.page_0['continue_btn_state'] = True
    elif span_start_input*2 < span_end_input:
        st.warning('Diffraction orders might overlap for the span selected. Might need order sorting filters. \
                    Do you wish to continue?', icon="⚠️")
        order_overlap_img = Image.open('img/order_overlap.png')
        st.image(order_overlap_img, "Diffraction orders overlap")
        st.session_state.page_0['continue_btn_state'] = False
    else:
        st.success("Lets move to the next section, please click continue button", icon="✔️")
        st.session_state.page_0['continue_btn_state'] = False

    continue_btn = st.button("Continue", help="Click to sumbit and navigate to next page", disabled=st.session_state.page_0['continue_btn_state'])

    if continue_btn:
        st.session_state.page_0 = {
            'spec_res_input': st.session_state.spectral_resolution,
            'span_start_input': st.session_state.span_start,
            'span_end_input': st.session_state.span_end
        }
        switch_page("Grating_specification")



