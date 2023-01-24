import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
# from numericalunits import nm, mm, cm, uM
from streamlit.components.v1 import html
from streamlit_extras.switch_page_button import switch_page
from grating import *
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.echo_expander import echo_expander


################################# functions ########################################
def calculated_optical_dispersion():
    blazewl_m_N = st.session_state.blaze_wavelength*(1e-6)* st.session_state.m * st.session_state.N
    m_n = (st.session_state.m*st.session_state.N)
    sin_alpha = np.sin(st.session_state.incident_angle*np.pi/180)

    return np.round(m_n/(np.sqrt(1-(blazewl_m_N-sin_alpha)**2)),2)

def calculated_incident_angle():
    blazewl_m_N = st.session_state.blaze_wavelength*(1e-6)* st.session_state.m * st.session_state.N
    sin_blaze_abgle = np.sin(st.session_state.blaze_angle*np.pi/180)

    cos_inv = np.arccos(blazewl_m_N/(2*sin_blaze_abgle))

    return np.round((cos_inv)*180/np.pi+st.session_state.blaze_angle,2)

def calcualte_beta(wavelength):

    m_lambda_N = st.session_state.m*wavelength*(1e-6)*st.session_state.N
    sin_alpha = np.sin(st.session_state.incident_angle*np.pi/180)

    return np.arcsin(m_lambda_N-sin_alpha)*180/np.pi
###################################################################################

st.session_state.update(st.session_state)
st.title(":blue[Angle, Line array & Lens Specification:]")
add_vertical_space(3)

user_input = st.container()

with st.sidebar:
    st.subheader("Incident angle, Refracted angle and Blaze angles")
    st.subheader("Line array Detector specifications")
    st.subheader("Lens focal length specification")

if 'page_4' not in st.session_state:
    st.session_state.page_4 = {
        'pixel_size_input' : 5.0,
        'incident_angle_input': 0.0
    }

st.session_state.incident_angle_calculated = calculated_incident_angle()

with user_input:

    st.info(f"With the selected Spectral resolution, Span, N value for the grating \
        calculated incident angle is _{st.session_state.incident_angle_calculated}_ degrees.\n \
            if you wish to change the incident angle please use below slider", icon="ℹ️")
    if np.isnan(st.session_state.incident_angle_calculated):
        st.error("caculated incident angle exeeds 90 degree, please set incident angle manually ")
        st.slider(label="_**Please chnage the incident angle here (in degrees)**_",
                min_value=0,
                max_value=200,
                value=0,
                key='incident_angle',
                step=1)
    else:
        st.slider(label="_**Please change the incident angle here (in degrees)**_",
                min_value=0,
                max_value=200,
                value=int(st.session_state.incident_angle_calculated),
                key='incident_angle',
                step=1)
    st.session_state.beta_min = np.round(calcualte_beta(st.session_state.span_start),2)
    st.session_state.beta_max = np.round(calcualte_beta(st.session_state.span_end),2)
    st.session_state.optical_span = st.session_state.beta_max-st.session_state.beta_min
    st.session_state.beta_middle = (st.session_state.beta_min + st.session_state.beta_max)/2

    add_vertical_space(1)

    with st.expander("Dispersion span of the system is:"):
        # st.info(f"Optical span of the system is \n\t", icon="ℹ️")
        st.markdown(f"###### Beta minimum = {st.session_state.beta_min}")
        st.markdown(f"###### Beta maximum = {st.session_state.beta_max}")
    add_vertical_space(1)

    st.number_input(label="_**Please enter the pixel size of the detector you want to use (in micrometer)**_",
                    min_value=1.0,
                    max_value=100.0,
                    value=st.session_state.page_4['pixel_size_input'],
                    key='pixel_size',
                    step=1.0)
    

    st.session_state.optical_dispersion = calculated_optical_dispersion()
    st.session_state.f2_calculated = np.round(1.5*st.session_state.pixel_size*1e-3/(np.tan(st.session_state.optical_dispersion*st.session_state.spectral_resolution*1e-6)/2),2)
    
    st.info(f"Prescibed values of f2: {st.session_state.f2_calculated}", icon="ℹ️")

    add_vertical_space(2)

    st.subheader("Change the value of F1 and F2 if you wish to")

    c1,c2 = st.columns(2)
    with c2:
        st.number_input(label="_**please set the value of f2:**_",
                        min_value=0.1,
                        max_value=200.0,
                        key='f2',
                        value=st.session_state.f2_calculated)
    with c1:
        st.number_input(label="_**please set the value of f1:**_",
                        min_value=0.1,
                        max_value=200.0,
                        key='f1',
                        value=50.0)

    st.session_state.sensor_width = np.round(2*st.session_state.f2*np.tan((st.session_state.beta_max-st.session_state.beta_min)*np.pi/180/2),2)
    st.session_state.slit_width = np.round((st.session_state.f1*3*st.session_state.pixel_size*1e-3)/st.session_state.f2,5)
    st.session_state.Optical_resolution = np.round((np.arctan(1.5*st.session_state.pixel_size*(1e-6)/st.session_state.f2)*2/st.session_state.optical_dispersion)*1e9,4)

    add_vertical_space(2)

    st.header("Ray tracing simulation:")

    with st.spinner('simulation running...'):
        design_rendering(f1=st.session_state.f1,
                            f2=st.session_state.f2,
                            lens_aperture=25.4,
                            grating_aperture=50.8,
                            sensor_width=st.session_state.sensor_width,
                            start_wavelength=st.session_state.span_start,
                            end_wavelength=st.session_state.span_end,
                            incident_angle=st.session_state.incident_angle,
                            difracted_angle=st.session_state.beta_middle,
                            N=st.session_state.N)
    
    if os.path.exists("img/grating.png"):
        image = Image.open("img/grating.png")
        st.image(image)

    add_vertical_space(2)

    st.header("Final design values:")

    st.subheader("User input:")
    a1, a2, a3 = st.columns(3)
    a1.metric(label="Spectral Resolution(nm)", value=st.session_state.spectral_resolution)
    a2.metric(label="Minimum wavelength(nm)", value=st.session_state.span_start)
    a3.metric(label="Maximum wavelength(nm)", value=st.session_state.span_end)
    style_metric_cards()

    st.subheader("Resolution:")
    f1, f2, f3 = st.columns(3)
    f1.metric(label="Resolution achieved (nm)", value=st.session_state.Optical_resolution)
    style_metric_cards()

    st.subheader("Grating specification:")
    b1, b2, b3 = st.columns(3)
    b1.metric(label="Grooves per mm", value=st.session_state.N)
    b2.metric(label="Blaze angle", value=f"{st.session_state.blaze_angle}°")
    style_metric_cards()

    st.subheader("Optical specification:")
    c1, c2 , c3 = st.columns(3)
    c1.metric(label="Incident angle", value=f"{st.session_state.incident_angle}°")
    c2.metric(label="Minimum difracted angle", value=f"{st.session_state.beta_min}°")
    c3.metric(label="Maximun difracted angle", value=f"{st.session_state.beta_max}°")
    style_metric_cards()

    d1, d2, d3 = st.columns(3)
    d1.metric(label="F1 in mm", value=np.round(st.session_state.f1,2))
    d2.metric(label="F2 in mm",value=st.session_state.f2)
    d3.metric(label="Slit width in mm", value=st.session_state.slit_width)
    style_metric_cards()

    st.subheader("Detector specification:")
    e1, e2, e3 = st.columns(3)
    e1.metric(label="Pixel size (in um)", value=st.session_state.pixel_size)
    e2.metric(label="Sensor width (in mm)", value=st.session_state.sensor_width)
    style_metric_cards(border_left_color="#FFC0CB", border_size_px=2)

    # st.text(st.session_state)



    
