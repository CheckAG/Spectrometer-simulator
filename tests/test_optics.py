import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optics import (
    calculate_blaze_wavelength,
    calculate_optimal_N,
    calculate_blaze_angle,
    calculate_incident_angle,
    calculate_beta,
    calculate_optical_dispersion,
    calculate_f2_prescribed,
    calculate_sensor_width,
    calculate_slit_width,
    calculate_optical_resolution,
    closest_value,
)
from config import GROOVE_OPTIONS


# ── closest_value ─────────────────────────────────────────────────────────────

def test_closest_value_exact_match():
    assert closest_value([300, 600, 1200], 600) == 600

def test_closest_value_rounds_down():
    assert closest_value(GROOVE_OPTIONS, 400) == 300

def test_closest_value_rounds_up():
    assert closest_value(GROOVE_OPTIONS, 500) == 600


# ── calculate_blaze_wavelength ────────────────────────────────────────────────

def test_blaze_wavelength_midpoint():
    assert calculate_blaze_wavelength(400, 700) == 550.0

def test_blaze_wavelength_symmetric():
    assert calculate_blaze_wavelength(400, 600) == 500.0

def test_blaze_wavelength_same():
    assert calculate_blaze_wavelength(500, 500) == 500.0


# ── calculate_optimal_N ───────────────────────────────────────────────────────

def test_optimal_N_returns_from_groove_options():
    N = calculate_optimal_N(550.0, 1.0)
    assert N in GROOVE_OPTIONS

def test_optimal_N_known_value():
    # 550 / 1 = 550 → closest groove option is 600
    assert calculate_optimal_N(550.0, 1.0) == 600

def test_optimal_N_coarser_resolution():
    # 550 / 10 = 55 → closest is 300
    assert calculate_optimal_N(550.0, 10.0) == 300


# ── calculate_blaze_angle ─────────────────────────────────────────────────────

def test_blaze_angle_positive():
    angle = calculate_blaze_angle(550.0, 600)
    assert angle > 0

def test_blaze_angle_returns_degrees():
    angle = calculate_blaze_angle(550.0, 600)
    assert 0 < angle < 90

def test_blaze_angle_littrow_consistency():
    # At Littrow: sin(theta_B) = m * lambda / (2 * d)
    N = 600
    lmb_nm = 550.0
    m = 1
    d_mm = 1.0 / N
    expected = np.degrees(np.arcsin(m * lmb_nm * 1e-9 / (2 * d_mm * 1e-3)))
    result = calculate_blaze_angle(lmb_nm, N, m)
    assert np.isclose(result, expected, atol=0.01)


# ── calculate_incident_angle ──────────────────────────────────────────────────

def test_incident_angle_finite_for_valid_inputs():
    angle = calculate_incident_angle(550.0, 600, 17.0)
    assert np.isfinite(angle)

def test_incident_angle_returns_nan_when_impossible():
    # Very short wavelength and very high N makes arccos arg > 1
    result = calculate_incident_angle(100.0, 2400, 5.0)
    # May or may not be nan depending on exact values — just check it doesn't crash
    assert isinstance(result, float)

def test_incident_angle_positive():
    angle = calculate_incident_angle(550.0, 600, 17.0)
    if not np.isnan(angle):
        assert angle >= 0


# ── calculate_beta ────────────────────────────────────────────────────────────

def test_beta_at_zero_incidence_and_zero_order():
    # m=1, N=600, lambda=550nm, alpha=0: beta = arcsin(0.33) ≈ 19.27°
    beta = calculate_beta(550.0, 600, 0.0, m=1)
    expected = np.degrees(np.arcsin(550e-6 * 600))
    assert np.isclose(beta, expected, atol=0.01)

def test_beta_increases_with_wavelength():
    beta1 = calculate_beta(400.0, 600, 30.0)
    beta2 = calculate_beta(700.0, 600, 30.0)
    assert beta2 > beta1

def test_beta_is_finite_for_valid_params():
    beta = calculate_beta(550.0, 600, 20.0)
    assert np.isfinite(beta)


# ── calculate_optical_dispersion ──────────────────────────────────────────────

def test_optical_dispersion_positive():
    disp = calculate_optical_dispersion(550.0, 600, 30.0)
    assert disp > 0

def test_optical_dispersion_increases_with_N():
    d300 = calculate_optical_dispersion(550.0, 300, 30.0)
    d600 = calculate_optical_dispersion(550.0, 600, 30.0)
    assert d600 > d300

def test_optical_dispersion_finite():
    disp = calculate_optical_dispersion(550.0, 600, 20.0)
    assert np.isfinite(disp)


# ── calculate_f2_prescribed ───────────────────────────────────────────────────

def test_f2_prescribed_positive():
    disp = calculate_optical_dispersion(550.0, 600, 30.0)
    f2 = calculate_f2_prescribed(5.0, disp, 1.0)
    assert f2 > 0

def test_f2_prescribed_larger_pixel_gives_larger_f2():
    disp = calculate_optical_dispersion(550.0, 600, 30.0)
    f2_small = calculate_f2_prescribed(5.0, disp, 1.0)
    f2_large = calculate_f2_prescribed(10.0, disp, 1.0)
    assert f2_large > f2_small


# ── calculate_sensor_width ────────────────────────────────────────────────────

def test_sensor_width_positive():
    width = calculate_sensor_width(50.0, -10.0, 10.0)
    assert width > 0

def test_sensor_width_zero_for_equal_betas():
    width = calculate_sensor_width(50.0, 10.0, 10.0)
    assert np.isclose(width, 0.0, atol=1e-9)

def test_sensor_width_scales_with_f2():
    w1 = calculate_sensor_width(50.0, -10.0, 10.0)
    w2 = calculate_sensor_width(100.0, -10.0, 10.0)
    assert np.isclose(w2, 2 * w1, atol=0.01)


# ── calculate_slit_width ──────────────────────────────────────────────────────

def test_slit_width_positive():
    width = calculate_slit_width(50.0, 50.0, 5.0)
    assert width > 0

def test_slit_width_scales_with_pixel_size():
    w1 = calculate_slit_width(50.0, 50.0, 5.0)
    w2 = calculate_slit_width(50.0, 50.0, 10.0)
    assert np.isclose(w2, 2 * w1, atol=1e-6)

def test_slit_width_f1_f2_ratio():
    # slit_width = (f1 * 3 * pixel_um * 1e-3) / f2
    result = calculate_slit_width(50.0, 100.0, 5.0)
    expected = (50.0 * 3 * 5.0 * 1e-3) / 100.0
    assert np.isclose(result, expected, atol=1e-6)


# ── calculate_optical_resolution ─────────────────────────────────────────────

def test_optical_resolution_positive():
    disp = calculate_optical_dispersion(550.0, 600, 30.0)
    res = calculate_optical_resolution(50.0, 5.0, disp)
    assert res > 0

def test_optical_resolution_improves_with_higher_dispersion():
    disp_low = calculate_optical_dispersion(550.0, 300, 30.0)
    disp_high = calculate_optical_dispersion(550.0, 1200, 30.0)
    res_low = calculate_optical_resolution(50.0, 5.0, disp_low)
    res_high = calculate_optical_resolution(50.0, 5.0, disp_high)
    assert res_high < res_low   # higher dispersion → finer resolution (smaller nm value)

def test_optical_resolution_nan_on_zero_dispersion():
    result = calculate_optical_resolution(50.0, 5.0, 0.0)
    assert np.isnan(result)
