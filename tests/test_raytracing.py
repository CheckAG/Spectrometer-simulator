import numpy as np
import pytest
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raytracing_v import (
    angle_wrap, propagate_rays,
    FocalElement, Lens, Grating, Sensor, Mirror, DMD, Aperture,
)


# ── angle_wrap ────────────────────────────────────────────────────────────────

def test_angle_wrap_identity_small():
    assert np.isclose(angle_wrap(0.5), 0.5)

def test_angle_wrap_above_pi():
    result = angle_wrap(np.pi + 0.1)
    assert np.isclose(result, -np.pi + 0.1, atol=1e-10)

def test_angle_wrap_below_neg_pi():
    result = angle_wrap(-np.pi - 0.1)
    assert np.isclose(result, np.pi - 0.1, atol=1e-10)

def test_angle_wrap_large_positive():
    result = angle_wrap(3 * np.pi)
    assert np.isclose(result, np.pi, atol=1e-10) or np.isclose(result, -np.pi, atol=1e-10)

def test_angle_wrap_large_negative():
    result = angle_wrap(-3 * np.pi)
    assert np.isclose(abs(result), np.pi, atol=1e-10)

def test_angle_wrap_accepts_array():
    angles = np.array([0.0, np.pi + 0.1, -np.pi - 0.1])
    result = angle_wrap(angles)
    assert result.shape == angles.shape
    assert np.all(np.abs(result) <= np.pi + 1e-10)


# ── FocalElement / Lens ───────────────────────────────────────────────────────

def test_lens_on_axis_ray_passes():
    lens = Lens(f=100, aperture=50, pos=[100, 0], theta=0)
    ray = np.array([0.0, 0.0, 0.0])          # on-axis, parallel
    dest = lens.propagate(ray)
    assert not np.isnan(dest[2])

def test_lens_blocks_ray_outside_aperture():
    lens = Lens(f=100, aperture=10, pos=[100, 0], theta=0)
    ray = np.array([0.0, 20.0, 0.0])         # y=20, beyond aperture/2=5
    dest = lens.propagate(ray)
    assert np.isnan(dest[2])

def test_focal_element_reflective_flag():
    mirror = FocalElement(f=100, aperture=50, pos=[100, 0], theta=0, reflective=True)
    assert mirror.reflective is True

def test_lens_alias():
    lens = Lens(f=50, aperture=25, pos=[50, 0], theta=0)
    assert isinstance(lens, FocalElement)
    assert lens.reflective is False


# ── Grating ───────────────────────────────────────────────────────────────────

def test_grating_transmissive_satisfies_diffraction_equation():
    N = 600
    lmb = 550e-9
    alpha = np.radians(20)
    grating = Grating(ngroves=N, aperture=50, pos=[0, 0], theta=0, transmissive=True)
    ray = np.array([0.0, 0.0, alpha])
    dest = grating.propagate(ray, lmb=lmb)
    if not np.isnan(dest[2]):
        d = 1e-3 / N
        expected_beta = np.arcsin(np.sin(alpha) - lmb / d)
        assert np.isclose(float(dest[2]), expected_beta, atol=1e-6)

def test_grating_reflective_beam_goes_backward():
    N = 600
    lmb = 550e-9
    alpha = np.radians(20)
    grating = Grating(ngroves=N, aperture=50, pos=[0, 0], theta=0, transmissive=False)
    ray = np.array([0.0, 0.0, alpha])
    dest = grating.propagate(ray, lmb=lmb)
    if not np.isnan(dest[2]):
        # Reflective: global angle should be in the backward hemisphere (|angle| > π/2)
        assert abs(float(dest[2])) > np.pi / 2

def test_grating_blocks_out_of_aperture():
    grating = Grating(ngroves=600, aperture=10, pos=[0, 0], theta=0, transmissive=True)
    ray = np.array([0.0, 20.0, 0.0])
    dest = grating.propagate(ray, lmb=550e-9)
    assert np.isnan(dest[2])


# ── Sensor ────────────────────────────────────────────────────────────────────

def test_sensor_terminates_ray():
    sensor = Sensor(aperture=50, pos=[200, 0], theta=0)
    ray = np.array([0.0, 0.0, 0.0])
    dest = sensor.propagate(ray)
    assert np.isnan(dest[2])


# ── Mirror ────────────────────────────────────────────────────────────────────

def test_mirror_reflects_normal_incidence():
    mirror = Mirror(aperture=50, pos=[100, 0], theta=0)
    ray = np.array([0.0, 0.0, 0.0])           # on-axis, angle=0
    dest = mirror.propagate(ray)
    # Normal incidence on theta=0 mirror: reflected angle = pi - 0 - 0 = pi (wrapped to -pi/pi)
    assert not np.isnan(dest[2])


# ── propagate_rays ────────────────────────────────────────────────────────────

def test_propagate_rays_output_shape():
    lens = Lens(f=100, aperture=50, pos=[100, 0], theta=0)
    sensor = Sensor(aperture=50, pos=[200, 0], theta=0)
    rays = [[0.0, y, 0.0] for y in np.linspace(-2, 2, 5)]
    bundles = propagate_rays([lens, sensor], rays, lmb=550e-9)
    assert bundles.shape == (3, 5, 3)   # (3-coords, 5-rays, 2-components+1)

def test_propagate_rays_first_column_is_initial():
    lens = Lens(f=100, aperture=50, pos=[100, 0], theta=0)
    rays = [[0.0, 0.0, 0.0]]
    bundles = propagate_rays([lens], rays, lmb=550e-9)
    assert np.allclose(bundles[:, 0, 0], [0.0, 0.0, 0.0])

def test_propagate_rays_no_figure_leak():
    """Each call to propagate_rays must not create matplotlib figures."""
    before = len(plt.get_fignums())
    lens = Lens(f=100, aperture=50, pos=[100, 0], theta=0)
    rays = [[0.0, 0.0, 0.0]]
    propagate_rays([lens], rays)
    after = len(plt.get_fignums())
    assert after == before


# ── tracer agrees with optics.py ─────────────────────────────────────────────

@pytest.mark.parametrize("transmissive,included", [(True, 0), (False, 30)])
def test_traced_spectrum_matches_analytic(transmissive, included):
    from optics import Spec, analyze
    from grating import trace
    spec = Spec(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048,
                f1=50, f2=50, included_deg=included, transmissive=transmissive)
    design = analyze(spec, 600)
    components, bundles = trace(spec, design, [spec.wl_min, spec.wl_max])
    sensor = components[-1]
    centres = []
    for b in bundles:
        pts = b[:2, :, -1]
        assert not np.isnan(pts).any()          # every ray reaches the sensor
        local = sensor.H @ np.vstack([pts, np.ones(pts.shape[1])])
        centres.append(local[1].mean())
    assert np.isclose(abs(centres[1] - centres[0]), design.spectrum_mm, rtol=1e-4)


# ── design_rendering ──────────────────────────────────────────────────────────

def test_design_rendering_returns_png_and_closes_figure():
    from optics import Spec, analyze
    from grating import design_rendering
    spec = Spec(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048,
                f1=50, f2=50)
    before = len(plt.get_fignums())
    png = design_rendering(spec, analyze(spec, 600))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(plt.get_fignums()) == before
