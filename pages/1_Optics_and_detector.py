import streamlit as st

from config import (
    DEFAULT_PIXEL_SIZE, DEFAULT_N_PIXELS, DEFAULT_SAMPLING, DEFAULT_F1, DEFAULT_F2,
    PIXEL_SIZE_MIN, PIXEL_SIZE_MAX, N_PIXELS_MIN, N_PIXELS_MAX,
    FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX, INCLUDED_ANGLE_MIN, INCLUDED_ANGLE_MAX,
)

if "page_0" not in st.session_state or "span_start" not in st.session_state.page_0:
    st.warning("Please complete the first page before continuing.")
    st.switch_page("User-input.py")

grating_type = st.session_state.page_0["grating_type"]
saved = st.session_state.get("page_1")
if saved is None or saved["grating_type"] != grating_type:
    saved = st.session_state.page_1 = {
        "grating_type": grating_type,
        "pixel_um": DEFAULT_PIXEL_SIZE, "n_pixels": DEFAULT_N_PIXELS,
        "sampling": DEFAULT_SAMPLING, "f1": DEFAULT_F1, "f2": DEFAULT_F2,
        "included_deg": None,  # None = computed for the chosen grating
    }

st.title(":blue[Detector & Optics]")
st.caption("The grating is chosen on the next page from these values.")

st.subheader("Line-array detector")
c1, c2, c3 = st.columns(3)
pixel_um = c1.number_input("Pixel pitch (µm)", PIXEL_SIZE_MIN, PIXEL_SIZE_MAX,
                           float(saved["pixel_um"]), step=0.5)
n_pixels = c2.number_input("Pixel count", N_PIXELS_MIN, N_PIXELS_MAX,
                           int(saved["n_pixels"]), step=1)
sampling = c3.number_input("Pixels per resolution element", 1.0, 10.0,
                           float(saved["sampling"]), step=0.5,
                           help="2 = Nyquist minimum, 3 = common practice.")
st.caption(f"Sensor length: {n_pixels * pixel_um / 1000:.2f} mm")

st.subheader("Lenses")
c1, c2 = st.columns(2)
f1 = c1.number_input("f1, collimator (mm)", FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX,
                     float(saved["f1"]))
f2 = c2.number_input("f2, focusing (mm)", FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX,
                     float(saved["f2"]),
                     help="Longer f2 = more dispersion on the sensor = finer resolution.")

st.subheader("Layout")
auto = st.toggle("Compute included angle automatically (recommended)",
                 value=saved["included_deg"] is None,
                 help="Transmissive: 0° (symmetric incidence, best efficiency). "
                      "Reflective: smallest angle at which M1 and M2 don't collide, + 5°.")
included_deg = None
if not auto:
    included_deg = st.slider(
        "Included angle α − β (°)", INCLUDED_ANGLE_MIN, INCLUDED_ANGLE_MAX,
        float(saved["included_deg"] or 0.0), step=1.0,
        help="Angle between the beam hitting the grating and the diffracted beam. "
             "0° = Littrow (reflective) or symmetric (transmissive).")

if st.button("Continue"):
    saved.update(pixel_um=pixel_um, n_pixels=int(n_pixels), sampling=sampling,
                 f1=f1, f2=f2, included_deg=included_deg)
    st.switch_page("pages/2_final_result.py")
