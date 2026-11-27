#!/usr/bin/env python3

import io

import numpy as np

from visualize import Canvas, get_colors
from ray_utilities import initial_rays
from raytracing_v import Lens, Grating, Sensor, propagate_rays
from config import N_WAVELENGTH_SAMPLES, N_RAYS, N_SOURCE_POINTS
from optics import Spec, Design


def build_components(spec: Spec, design: Design):
    """
    Layout: slit at origin → F1 at x=f1 → grating at x=2·f1 → F2 and sensor
    along the centre diffracted ray. Returns (components, slit_y_points).
    """
    f1, f2 = spec.f1, spec.f2
    alpha, beta = np.radians(design.alpha), np.radians(design.beta_center)

    # Direction (global) of the centre diffracted ray. The tracer measures β
    # with the opposite sign to optics.py, hence these forms.
    place_dir = -(alpha + beta) if spec.transmissive else np.pi + beta - alpha
    theta_elem = -place_dir  # elements face the incoming centre ray

    x1, y1 = 2 * f1 + f2 * np.cos(place_dir), f2 * np.sin(place_dir)
    x2, y2 = x1 + f2 * np.cos(place_dir), y1 + f2 * np.sin(place_dir)

    components = [
        Lens(f=f1, aperture=spec.beam_mm, pos=[f1, 0], theta=0, name="F1"),
        Grating(ngroves=design.G, aperture=spec.grating_mm, pos=[2 * f1, 0],
                theta=alpha, m=spec.m, transmissive=spec.transmissive),
        Lens(f=f2, aperture=spec.grating_mm, pos=[x1, y1], theta=theta_elem, name="F2"),
        Sensor(aperture=spec.sensor_mm, pos=[x2, y2], theta=theta_elem),
    ]
    half_slit = design.slit_um * 1e-3 / 2
    return components, np.linspace(half_slit, -half_slit, N_SOURCE_POINTS)


def trace(spec: Spec, design: Design, wavelengths_nm):
    """Ray bundles (see propagate_rays) for each wavelength."""
    components, slit_y = build_components(spec, design)
    scene = np.zeros((2, len(slit_y)))
    scene[1, :] = slit_y
    rays, _, _ = initial_rays(scene, components[0], N_RAYS)
    return components, [propagate_rays(components, rays, lmb=wl * 1e-9)
                        for wl in wavelengths_nm]


def design_rendering(spec: Spec, design: Design) -> bytes:
    """2-D ray-trace drawing of the design as PNG bytes (no files written)."""
    wavelengths = np.linspace(spec.wl_min, spec.wl_max, N_WAVELENGTH_SAMPLES)
    components, bundles = trace(spec, design, wavelengths)

    xs = [c.pos[0] for c in components]
    ys = [c.pos[1] for c in components]
    margin = spec.grating_mm * 2
    y_margin = max(spec.f1, max(abs(y) for y in ys)) * 1.5
    canvas = Canvas(xlim=[min(xs) - margin, max(xs) + margin], ylim=[-y_margin, y_margin])
    canvas.draw_components(components)

    colors = get_colors(len(wavelengths), N_RAYS * N_SOURCE_POINTS, cmap="rainbow")
    for color, bundle in zip(colors, bundles):
        canvas.draw_rays(bundle, color, linewidth=0.2)

    buf = io.BytesIO()
    canvas.save(buf)
    canvas.close()
    return buf.getvalue()
