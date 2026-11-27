"""Layouts traced end to end, checked against the analytic design (optics.py)."""

import numpy as np
import pytest

from spectrometer.layout import trace, sensor_hits, aberration_blur
from spectrometer.optics import Spec, analyze, sensor_position, detector_signal, line_dip
from spectrometer.plot import ray_figure

def _spec(**kw):
    base = dict(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048,
                f1=50, f2=50)
    return Spec(**{**base, **kw})


@pytest.mark.parametrize("transmissive,included", [(True, 0), (True, 20), (False, 40), (False, 60)])
@pytest.mark.parametrize("G", [300, 600])
def test_traced_spectrum_matches_analytic(transmissive, included, G):
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
    spec = _spec(wl_min=400, wl_max=900, included_deg=included, transmissive=transmissive)
    design = analyze(spec, 300)
    c, first = trace(spec, design, [800])
    _, second = trace(spec, design, [400], m=2)
    assert np.isclose(np.nanmean(sensor_hits(c, first)), np.nanmean(sensor_hits(c, second)))


# ── plot ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("transmissive", [True, False])
def test_ray_figure_has_every_order_and_element(transmissive):
    spec = _spec(wl_min=400, wl_max=900, transmissive=transmissive)
    fig = ray_figure(spec, analyze(spec, 300))
    legend = {t.name for t in fig.data if t.showlegend is not False}
    assert {n.split()[2] for n in legend} == {"0", "1", "2"}      # "m = k ..."
    labels = {a.text for a in fig.layout.annotations}
    names = {"<b>F1</b>", "<b>F2</b>"} if transmissive else {"<b>M1</b>", "<b>M2</b>"}
    assert names | {"<b>Grating</b>", "<b>Sensor</b>"} <= labels


@pytest.mark.parametrize("transmissive,angle", [(True, 0.0), (False, 40.0)])
def test_sensor_position_matches_trace(transmissive, angle):
    """Analytic wavelength → detector mapping agrees with where traced rays land."""
    spec = _spec(transmissive=transmissive, included_deg=angle)
    design = analyze(spec, 300)
    wls = np.array([400.0, 480.0, 550.0, 620.0, 700.0])
    components, bundles = trace(spec, design, wls)
    hits = np.nanmean(sensor_hits(components, bundles).reshape(len(wls), -1), axis=1)
    x = sensor_position(spec, design, wls)
    assert abs(x[2]) < 1e-9                        # centre wavelength at the sensor centre
    assert np.allclose(np.abs(hits - hits[2]), np.abs(x), atol=1e-3)


def test_detector_signal_peaks_and_doublet():
    spec = _spec()
    design = analyze(spec, 300)                    # resolution ≈ 1 nm
    wl, sig = detector_signal(spec, design, [450.0, 650.0])
    for line in (450.0, 650.0):
        near = np.abs(wl - line) < 5
        assert abs(wl[near][np.argmax(sig[near])] - line) < design.rld * spec.pixel_um * 1e-3
    split = lambda sep: line_dip(*detector_signal(spec, design, [577, 577 + sep]), 577, 577 + sep)
    assert split(2 * design.resolution) < 0.81 < split(0.6 * design.resolution)
    assert np.isnan(line_dip(wl, sig, 100, 110))     # off the sensor



def test_aberration_blur_scales_with_beam_and_vanishes_for_lenses():
    blur = lambda **kw: aberration_blur(_spec(**kw), analyze(_spec(**kw), 300))[1]
    assert not blur(transmissive=True).any()                # ideal lenses: no aberration
    wide, narrow = blur(transmissive=False), blur(transmissive=False, beam_mm=6.35)
    assert np.all(narrow < wide / 5)                         # coma ∝ y², spherical ∝ y³
    slow = blur(transmissive=False, beam_mm=6.35, f1=200, f2=200, n_pixels=4096)
    assert np.all(slow < 0.1)                                # slow, barely folded: near ideal
