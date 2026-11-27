import numpy as np
import pytest
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


# ── tracer agrees with optics.py ─────────────────────────────────────────────

def _spec(**kw):
    from optics import Spec
    base = dict(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048,
                f1=50, f2=50)
    return Spec(**{**base, **kw})


@pytest.mark.parametrize("transmissive,included", [(True, 0), (True, 20), (False, 40), (False, 60)])
@pytest.mark.parametrize("G", [300, 600])
def test_traced_spectrum_matches_analytic(transmissive, included, G):
    from optics import analyze
    from grating import trace, sensor_hits
    spec = _spec(included_deg=included, transmissive=transmissive, n_pixels=4096)
    design = analyze(spec, G)
    assert design.spectrum_mm <= spec.sensor_mm
    components, bundles = trace(spec, design, [spec.wl_min, spec.wl_max])
    y = sensor_hits(components, bundles).reshape(2, -1)
    assert not np.isnan(y).any()                       # every ray lands on the sensor
    assert np.isclose(abs(y[1].mean() - y[0].mean()), design.spectrum_mm, rtol=1e-4)
    # Slit image spans `sampling` pixels (ideal optics: no aberration blur).
    assert np.allclose(np.ptp(y, axis=1), spec.sampling * spec.pixel_um * 1e-3, rtol=0.15)


@pytest.mark.parametrize("transmissive,included", [(True, 0), (False, 40)])
def test_second_order_overlaps_first_order_at_double_wavelength(transmissive, included):
    from optics import analyze
    from grating import trace, sensor_hits
    spec = _spec(wl_min=400, wl_max=900, included_deg=included, transmissive=transmissive)
    design = analyze(spec, 300)
    c, first = trace(spec, design, [800])
    _, second = trace(spec, design, [400], m=2)
    assert np.isclose(np.nanmean(sensor_hits(c, first)), np.nanmean(sensor_hits(c, second)))


# ── plot ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("transmissive", [True, False])
def test_ray_figure_has_every_order_and_element(transmissive):
    from optics import analyze
    from plot import ray_figure
    spec = _spec(wl_min=400, wl_max=900, transmissive=transmissive)
    fig = ray_figure(spec, analyze(spec, 300))
    legend = {t.name for t in fig.data if t.showlegend}
    assert {n.split()[2] for n in legend} == {"0", "1", "2"}      # "m = k ..."
    labels = {a.text for a in fig.layout.annotations}
    names = {"<b>F1</b>", "<b>F2</b>"} if transmissive else {"<b>M1</b>", "<b>M2</b>"}
    assert names | {"<b>Grating</b>", "<b>Sensor</b>"} <= labels
