import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optics import (
    Spec, analyze, angular_dispersion, blaze_angle, default_blaze_wavelength,
    diffraction_angle, grating_angles, recommend_groove_density,
)


def spec(**kw):
    base = dict(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048,
                f1=50, f2=50)
    return Spec(**{**base, **kw})


# ── grating equation ──────────────────────────────────────────────────────────

def test_diffraction_angle_satisfies_grating_equation():
    beta = diffraction_angle(550, 600, 20, m=1)
    assert np.isclose(550e-6 * 600, np.sin(np.radians(20)) + np.sin(np.radians(beta)))


def test_diffraction_angle_vectorised_and_increasing():
    betas = diffraction_angle([400, 550, 700], 600, 20)
    assert betas.shape == (3,) and np.all(np.diff(betas) > 0)


def test_evanescent_order_is_nan():
    assert np.isnan(diffraction_angle(1500, 2400, 0))


def test_littrow_textbook_value():
    # 1200 l/mm, 500 nm, first-order Littrow: α = β = 17.46°
    alpha, beta = grating_angles(500, 1200, included_deg=0)
    assert np.isclose(alpha, 17.46, atol=0.01) and np.isclose(alpha, beta)


@pytest.mark.parametrize("included", [0, 15, 30, 60])
def test_grating_angles_honour_included_angle(included):
    alpha, beta = grating_angles(550, 600, included)
    assert np.isclose(alpha - beta, included)
    assert np.isclose(diffraction_angle(550, 600, alpha), beta)


def test_angular_dispersion_matches_numerical_derivative():
    alpha = 20
    h = 1e-3
    num = np.radians(diffraction_angle(550 + h, 600, alpha)
                     - diffraction_angle(550 - h, 600, alpha)) / (2 * h)
    beta = diffraction_angle(550, 600, alpha)
    assert np.isclose(angular_dispersion(600, beta), num, rtol=1e-6)


# ── blaze ─────────────────────────────────────────────────────────────────────

def test_reflective_littrow_blaze_angle():
    expected = np.degrees(np.arcsin(500e-6 * 1200 / 2))
    assert np.isclose(blaze_angle(500, 1200, 0), expected)


def test_transmissive_blaze_satisfies_facet_refraction():
    n = 1.5
    theta = np.radians(blaze_angle(500, 600, 0, transmissive=True, n_glass=n))
    alpha, beta = grating_angles(500, 600, 0)
    delta = np.radians(alpha + beta)
    assert np.isclose(n * np.sin(theta), np.sin(theta + delta))


def test_default_blaze_window_covers_visible_span():
    wl_b = default_blaze_wavelength(400, 700)
    assert 0.67 * wl_b <= 400 and 700 <= 1.8 * wl_b


# ── analyze ───────────────────────────────────────────────────────────────────

def test_resolution_scales_inversely_with_f2():
    a, b = analyze(spec(f2=50), 600), analyze(spec(f2=100), 600)
    assert np.isclose(a.resolution_image, 2 * b.resolution_image, rtol=0.01)


def test_slit_image_matches_sampling_pixels():
    d = analyze(spec(f1=50, f2=100, included_deg=0), 600)
    # Symmetric layout: cos α / cos β = 1, so magnification = f2/f1 = 2
    assert np.isclose(d.slit_um * 2, 3 * 5.0)


def test_grating_limit_from_illuminated_grooves():
    s = spec()
    d = analyze(s, 600)
    footprint = s.beam_mm / np.cos(np.radians(d.alpha))
    assert np.isclose(d.resolution_grating, s.wl_center / (600 * footprint))


def test_too_few_pixels_warns():
    d = analyze(spec(resolution=0.1, n_pixels=512), 600)
    assert any("pixels" in w for w in d.warnings)


def test_order_overlap_warns():
    d = analyze(spec(wl_min=400, wl_max=900), 300)
    assert any("Second order" in w for w in d.warnings)


def test_impossible_geometry_is_not_ok():
    d = analyze(spec(wl_min=1000, wl_max=1900, included_deg=60), 2400)
    assert not d.ok


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommendation_meets_target_and_fits_sensor():
    s = spec()
    d = recommend_groove_density(s)
    assert d.resolution <= s.resolution and d.spectrum_mm <= s.sensor_mm


def test_finer_target_needs_more_lines():
    coarse = recommend_groove_density(spec(resolution=2.0))
    fine = recommend_groove_density(spec(resolution=0.3, n_pixels=4096))
    assert fine.G > coarse.G


def test_ct_mirror_overlap_warns_only_when_tight():
    tight = analyze(spec(transmissive=False, included_deg=10), 300)
    roomy = analyze(spec(transmissive=False, included_deg=60), 300)
    assert any("overlap" in w for w in tight.warnings)
    assert not any("overlap" in w for w in roomy.warnings)


def test_ct_detector_tilt_stretches_spectrum():
    flat = analyze(spec(transmissive=True), 300)
    ct = analyze(spec(transmissive=False, included_deg=0), 300)
    assert ct.fold2 > 0 and ct.spectrum_mm > flat.spectrum_mm


# ── auto included angle ───────────────────────────────────────────────────────

def test_auto_angle_transmissive_is_symmetric():
    d = analyze(spec(transmissive=True), 300)
    assert d.included_deg == 0 and np.isclose(d.alpha, d.beta_center)


def test_auto_angle_reflective_is_minimum_clearing_plus_margin():
    from config import AUTO_ANGLE_MARGIN
    d = analyze(spec(transmissive=False), 300)
    assert d.mirrors_clear and not d.warnings
    just_below = analyze(spec(transmissive=False,
                              included_deg=d.included_deg - AUTO_ANGLE_MARGIN - 1), 300)
    assert not just_below.mirrors_clear


def test_auto_angle_shrinks_with_longer_focal_length():
    short = analyze(spec(transmissive=False, f1=50, f2=50), 300)
    long = analyze(spec(transmissive=False, f1=150, f2=150), 300)
    assert long.included_deg < short.included_deg


def test_recommendation_when_nothing_fits_picks_closest_to_fitting():
    s = spec(f1=150, f2=150)            # even 300 l/mm overflows a 10 mm sensor
    d = recommend_groove_density(s)
    assert d.G == 300 and d.spectrum_mm > s.sensor_mm


# ── one-click fixes ───────────────────────────────────────────────────────────

def _apply(s, fix):
    from dataclasses import replace
    return replace(s, **fix)


@pytest.mark.parametrize("kw,G,field", [
    (dict(f1=150, f2=150), 300, "f2"),                         # spectrum overflows sensor
    (dict(f1=150, f2=150, transmissive=False), 300, "f2"),     # same, Czerny–Turner (nonlinear)
    (dict(resolution=0.1, n_pixels=512), 600, "n_pixels"),     # too few pixels
    (dict(resolution=0.3), 300, "resolution"),                 # target not met
    (dict(wl_min=400, wl_max=900), 300, "wl_max"),             # order overlap
    (dict(transmissive=False, included_deg=10), 300, "included_deg"),  # mirrors collide
])
def test_every_fix_clears_its_warning(kw, G, field):
    s = spec(**kw)
    issues = [w for w in analyze(s, G).warnings if w.field == field and w.fixes]
    assert issues
    for issue in issues:
        for label, fix in issue.fixes.items():
            after = analyze(_apply(s, fix), G).warnings
            assert not any(w.field == field and w[:20] == issue[:20] for w in after), label
