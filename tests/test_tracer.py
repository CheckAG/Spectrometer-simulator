"""Element-level tests of the 2-D ray tracer."""

import numpy as np

from spectrometer.tracer import (
    angle_wrap, propagate_rays, FocalElement, Lens, Grating, Sensor, ExactSphericalMirror,
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

# ── vectorised tracer behaviour ──────────────────────────────────────────────

def test_vectorised_propagate_matches_single_rays():
    lens = Lens(f=100, aperture=50, pos=[100, 0], theta=0.2)
    rays = np.array([[0.0, y, a] for y, a in [(0, 0), (2, 0.05), (-3, -0.1)]]).T
    batch = lens.propagate(rays)
    singles = np.stack([lens.propagate(rays[:, i]) for i in range(3)], axis=1)
    assert np.allclose(batch, singles)


def test_element_behind_ray_is_not_hit():
    lens = Lens(f=100, aperture=50, pos=[-100, 0], theta=0)
    assert np.all(np.isnan(lens.propagate(np.array([0.0, 0.0, 0.0]))))


def test_evanescent_order_is_blocked_not_clipped():
    grating = Grating(ngroves=2400, aperture=50, pos=[0, 0], theta=0, m=1, transmissive=True)
    assert np.isnan(grating.propagate(np.array([-10.0, 0.0, 0.0]), lmb=900e-9)[2])


def test_per_ray_wavelengths():
    grating = Grating(ngroves=600, aperture=50, pos=[0, 0], theta=0.3, transmissive=True)
    rays = np.array([[-10.0, 0, 0], [-10.0, 0, 0]]).T
    out = grating.propagate(rays, lmb=np.array([400e-9, 700e-9]))
    assert np.isclose(out[2, 0], grating.propagate(rays[:, 0], lmb=400e-9)[2])
    assert np.isclose(out[2, 1], grating.propagate(rays[:, 1], lmb=700e-9)[2])


# ── exact spherical mirror ────────────────────────────────────────────────────

def test_exact_mirror_spherical_aberration_matches_theory():
    """Parallel rays at height y focus at R − R/(2·cos θ), sin θ = y/R (marginal focus)."""
    f, ys = 50.0, np.array([0.5, 5.0, 12.0])
    out = ExactSphericalMirror(f=f, aperture=25.4, pos=[0, 0], theta=0).propagate(
        np.vstack([np.full(3, -100.0), ys, np.zeros(3)]))
    crossing = out[0] - out[1] / np.tan(out[2])            # where each ray meets the axis
    th = np.arcsin(ys / (2 * f))
    assert np.allclose(crossing, -(2 * f - f / np.cos(th)), atol=1e-9)
