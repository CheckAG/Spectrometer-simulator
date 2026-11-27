#!/usr/bin/env python3

import io

import numpy as np

from visualize import Canvas, get_colors
from ray_utilities import initial_rays
from raytracing_v import Lens, SphericalMirror, Grating, Sensor, propagate_rays
from config import N_WAVELENGTH_SAMPLES, N_RAYS, N_SOURCE_POINTS, EXTRA_ORDERS
from optics import Spec, Design


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def build_components(spec: Spec, design: Design, m=None):
    """
    Transmissive: 4f inline, slit → F1 → grating → F2 → sensor.
    Reflective: Czerny–Turner, slit → M1 → grating → M2 → sensor.
    `m` overrides the grating order (to trace stray orders).
    Returns (components, slit_points (2, N_SOURCE_POINTS)).
    """
    f1, f2 = spec.f1, spec.f2
    alpha, beta = np.radians(design.alpha), np.radians(design.beta_center)
    m = spec.m if m is None else m

    if spec.transmissive:
        # Tracer measures β with the opposite sign to optics.py, hence −(α + β).
        out_dir = -(alpha + beta)
        theta_elem = -out_dir                     # F2 and sensor face the centre ray
        p_f2 = np.array([2 * f1, 0]) + f2 * _unit(out_dir)
        p_sensor = p_f2 + f2 * _unit(out_dir)
        components = [
            Lens(f=f1, aperture=spec.beam_mm, pos=[f1, 0], theta=0, name="F1"),
            Grating(ngroves=design.G, aperture=spec.grating_mm, pos=[2 * f1, 0],
                    theta=alpha, m=m, transmissive=True),
            Lens(f=f2, aperture=spec.grating_mm, pos=p_f2, theta=theta_elem, name="F2"),
            Sensor(aperture=spec.sensor_mm, pos=p_sensor, theta=theta_elem),
        ]
        slit_pos, slit_dir = np.zeros(2), 0.0
    else:
        # Czerny–Turner: grating at the origin, normal pointing down (−y).
        # M1 sits at angle α from the normal (right), M2 at β (left for β < 0).
        # Tracer: grating theta = −90° makes local incidence = α for a beam
        # arriving from M1, and the diffracted centre ray heads to M2.
        to_m1, to_m2 = alpha - np.pi / 2, beta - np.pi / 2
        p_m1, p_m2 = f1 * _unit(to_m1), f2 * _unit(to_m2)

        def fold(f, pos, psi_in, off, aperture, name):
            """
            Ideal spherical mirror at `pos` hit by a chief ray travelling at
            psi_in, at local incidence `off`; it leaves at π + psi_in − 2·off.
            Returns (mirror, point on its focal plane along the chief ray).
            """
            mirror = SphericalMirror(f=f, aperture=aperture, pos=pos,
                                     theta=off - psi_in, name=name)
            fx, fy = mirror.to_global(np.array([-f]), np.array([f * np.tan(off)]))
            return mirror, np.array([fx[0], fy[0]])

        # Negative off-axis angles fold the beam outward, away from the grating.
        # ponytail: ideal mirrors; real off-axis coma/astigmatism needs exact surfaces
        off1, off2 = -np.radians(design.fold1), -np.radians(design.fold2)

        # M1: choose the incoming direction so the reflection heads to the grating.
        slit_dir = to_m1 + 2 * off1               # (to_m1 + π) − π + 2·off1
        m1, _ = fold(f1, p_m1, slit_dir, off1, spec.beam_mm, "M1")
        # Slit sits on M1's front focal plane (ideal element: local x = −f1).
        slit_pos = p_m1 - f1 / np.cos(off1) * _unit(slit_dir)

        # Beam widens by cos β / cos α at the grating, then walks with λ.
        beam2 = spec.beam_mm * np.cos(beta) / np.cos(alpha)
        spread = f2 * abs(np.radians(design.beta_max - design.beta_min))
        m2, sensor_pos = fold(f2, p_m2, to_m2, off2, (beam2 + spread) / np.cos(off2), "M2")
        components = [
            m1,
            Grating(ngroves=design.G, aperture=spec.grating_mm, pos=[0, 0],
                    theta=-np.pi / 2, m=m, transmissive=False),
            m2,
            Sensor(aperture=spec.sensor_mm, pos=sensor_pos, theta=m2.theta),
        ]

    # Slit: short line perpendicular to the chief ray leaving it.
    half_slit = design.slit_um * 1e-3 / 2
    across = _unit(slit_dir + np.pi / 2)
    slit = slit_pos[:, None] + across[:, None] * np.linspace(half_slit, -half_slit, N_SOURCE_POINTS)
    return components, slit


def trace(spec: Spec, design: Design, wavelengths_nm, m=None):
    """
    Trace every wavelength in one call.
    Returns (components, bundles) with bundles shaped (3, n_wl·n_rays, n_comp+1),
    rays grouped by wavelength (wavelength i = rays i·n_rays … (i+1)·n_rays − 1).
    """
    components, slit = build_components(spec, design, m)
    rays, _, _ = initial_rays(slit, components[0], N_RAYS)
    n = len(rays)
    wl = np.repeat(np.asarray(wavelengths_nm, dtype=float) * 1e-9, n)
    return components, propagate_rays(components, np.tile(rays, (len(wavelengths_nm), 1)), lmb=wl)


def sensor_hits(components, bundles):
    """Local positions (mm) on the sensor; NaN for rays that miss it."""
    sensor = components[-1]
    pts = bundles[:2, :, -1]
    y = (sensor.H @ np.vstack([pts, np.ones(pts.shape[1])]))[1]
    return np.where(np.abs(y) <= sensor.aperture / 2, y, np.nan)


def design_rendering(spec: Spec, design: Design, extra_orders=EXTRA_ORDERS) -> bytes:
    """
    2-D ray-trace drawing of the design as PNG bytes (no files written).
    The design order is drawn solid; `extra_orders` (e.g. 0 and 2) faint, to
    show stray light and order overlap.
    """
    wavelengths = np.linspace(spec.wl_min, spec.wl_max, N_WAVELENGTH_SAMPLES)
    n_rays = N_RAYS * N_SOURCE_POINTS
    colors = np.repeat(np.array(get_colors(len(wavelengths), 1, cmap="rainbow"))[:, 0], n_rays, axis=0)
    components, bundles = trace(spec, design, wavelengths)

    xs = [c.pos[0] for c in components] + [0]
    ys = [c.pos[1] for c in components] + [0]
    margin = spec.grating_mm
    canvas = Canvas(xlim=[min(xs) - margin, max(xs) + margin],
                    ylim=[min(ys) - margin, max(ys) + margin])
    canvas.draw_components(components)

    for m in extra_orders:
        if m == spec.m:
            continue
        _, stray = trace(spec, design, wavelengths, m=m)
        canvas.draw_rays(stray, np.c_[colors, np.full(len(colors), 0.25)], linewidth=0.2,
                         linestyle="--")
    canvas.draw_rays(bundles, colors, linewidth=0.3)

    buf = io.BytesIO()
    canvas.save(buf)
    return buf.getvalue()
