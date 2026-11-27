#!/usr/bin/env python3

import numpy as np

from ray_utilities import initial_rays
from raytracing_v import Lens, SphericalMirror, ExactSphericalMirror, Grating, Sensor, propagate_rays
from config import N_RAYS, N_SOURCE_POINTS
from optics import Spec, Design


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def build_components(spec: Spec, design: Design, m=None, exact=False):
    """
    Transmissive: 4f inline, slit → F1 → grating → F2 → sensor.
    Reflective: Czerny–Turner, slit → M1 → grating → M2 → sensor.
    `m` overrides the grating order (to trace stray orders).
    `exact` (reflective only): real spherical mirrors. The slit then sits at
    M1's tangential focus (f1·cos fold1) and the sensor on the traced
    best-focus line, as one would align a real instrument.
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

        Mirror = ExactSphericalMirror if exact else SphericalMirror

        def fold(f, pos, psi_in, off, aperture, name):
            """
            Ideal spherical mirror at `pos` hit by a chief ray travelling at
            psi_in, at local incidence `off`; it leaves at π + psi_in − 2·off.
            Returns (mirror, point on its focal plane along the chief ray).
            """
            mirror = Mirror(f=f, aperture=aperture, pos=pos,
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
        # Exact sphere: off-axis (tangential) focus is at f·cos(off) (Coddington).
        slit_pos = p_m1 - (f1 * np.cos(off1) if exact else f1 / np.cos(off1)) * _unit(slit_dir)

        # Beam widens by cos β / cos α at the grating, then walks with λ.
        beam2 = spec.beam_mm * np.cos(beta) / np.cos(alpha)
        spread = f2 * abs(np.radians(design.beta_max - design.beta_min))
        m2, sensor_pos = fold(f2, p_m2, to_m2, off2, (beam2 + spread) / np.cos(off2), "M2")
        grating = Grating(ngroves=design.G, aperture=spec.grating_mm, pos=[0, 0],
                          theta=-np.pi / 2, m=m, transmissive=False)
        sensor_theta = m2.theta
        if exact:
            # Sensor through the best foci of the span ends, centred on the centre focus.
            # Design order always, so stray orders hit the same sensor.
            g1 = Grating(ngroves=design.G, aperture=spec.grating_mm, pos=[0, 0],
                         theta=-np.pi / 2, m=spec.m, transmissive=False)
            foci = [_best_focus([m1, g1, m2], slit_pos, wl)
                    for wl in (spec.wl_min, spec.wl_center, spec.wl_max)]
            along = foci[2] - foci[0]
            sensor_pos, sensor_theta = foci[1], np.pi / 2 - np.arctan2(along[1], along[0])
        components = [m1, grating, m2,
                      Sensor(aperture=spec.sensor_mm, pos=sensor_pos, theta=sensor_theta)]

    # Slit: short line perpendicular to the chief ray leaving it.
    half_slit = design.slit_um * 1e-3 / 2
    across = _unit(slit_dir + np.pi / 2)
    slit = slit_pos[:, None] + across[:, None] * np.linspace(half_slit, -half_slit, N_SOURCE_POINTS)
    return components, slit


def _best_focus(components, point, wl_nm, n_rays=101):
    """
    Point closest (least squares) to every ray of a fan from `point` that
    leaves the last component: the circle of least confusion in 2-D.
    """
    rays, _, _ = initial_rays(np.asarray(point, float)[:, None], components[0], n_rays)
    out = propagate_rays(components, rays, lmb=wl_nm * 1e-9)[:, :, -1]
    out = out[:, np.isfinite(out).all(axis=0)]
    if out.shape[1] < 2:                                      # nothing gets through
        return np.full(2, np.nan)
    u = np.vstack([np.cos(out[2]), np.sin(out[2])])          # (2, N)
    P = np.eye(2)[None] - np.einsum("in,jn->nij", u, u)       # projectors ⟂ each ray
    return np.linalg.solve(P.sum(0), np.einsum("nij,jn->i", P, out[:2]))


def aberration_blur(spec: Spec, design: Design, n_wl=9, n_rays=201):
    """
    Spectral blur (nm, FWHM-equivalent 2.355·σ) that the mirrors add, per
    wavelength: a point on the slit traced through exact mirrors, its RMS
    spot width on the sensor divided by the traced linear dispersion.
    Returns (wavelengths_nm, blur_nm). Transmissive layouts use ideal lenses: zeros.
    """
    wls = np.linspace(spec.wl_min, spec.wl_max, n_wl)
    if spec.transmissive:
        return wls, np.zeros(n_wl)
    components, slit = build_components(spec, design, exact=True)
    rays, _, _ = initial_rays(slit.mean(axis=1)[:, None], components[0], n_rays)
    bundles = propagate_rays(components, np.tile(rays, (n_wl, 1)), lmb=np.repeat(wls * 1e-9, n_rays))
    pts = bundles[:2, :, -1]                                  # unclipped: blur even off the sensor
    hits = (components[-1].H @ np.vstack([pts, np.ones(pts.shape[1])]))[1].reshape(n_wl, n_rays)
    if np.isnan(hits).all():
        return wls, np.full(n_wl, np.nan)
    centre = np.nanmean(hits, axis=1)
    dx_dl = np.gradient(centre, wls)                          # mm/nm along the sensor
    with np.errstate(invalid="ignore", divide="ignore"):
        return wls, 2.355 * np.nanstd(hits, axis=1) / np.abs(dx_dl)


def trace(spec: Spec, design: Design, wavelengths_nm, m=None, exact=False):
    """
    Trace every wavelength in one call.
    Returns (components, bundles) with bundles shaped (3, n_wl·n_rays, n_comp+1),
    rays grouped by wavelength (wavelength i = rays i·n_rays … (i+1)·n_rays − 1).
    """
    components, slit = build_components(spec, design, m, exact)
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
