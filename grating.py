#!/usr/bin/env python3

import os
import numpy as np

from visualize import Canvas, get_colors
from ray_utilities import initial_rays
from raytracing_v import angle_wrap, Lens, Grating, Sensor, propagate_rays
from config import (N_WAVELENGTH_SAMPLES, N_RAYS, N_SOURCE_POINTS,
                    LENS_APERTURE, GRATING_APERTURE, SIMULATION_OUTPUT_PATH)


def _reflective_theta_design(incident_angle_deg: float, N: int,
                              start_wavelength_nm: float, end_wavelength_nm: float) -> float:
    """
    Compute the direction (global frame, radians) of the central diffracted ray
    for a reflective grating layout.

    The grating equation in local frame: sin(β_local) = sin(α_local) − mλ/d
    After reflective bounce the global angle = π − β_local − grating_theta.
    """
    lmb_center = (start_wavelength_nm + end_wavelength_nm) / 2.0 * 1e-9  # metres
    d = 1e-3 / N                                                           # metres
    inc_rad = np.radians(incident_angle_deg)
    sin_local = np.clip(np.sin(inc_rad) - lmb_center / d, -1.0, 1.0)
    local_ref = np.arcsin(sin_local)
    return float(angle_wrap(np.pi - local_ref - inc_rad))


def design_rendering(f1, f2, sensor_width, start_wavelength, end_wavelength,
                     incident_angle, difracted_angle, N,
                     reflective: bool = False,
                     lens_aperture=LENS_APERTURE,
                     grating_aperture=GRATING_APERTURE,
                     output_path=SIMULATION_OUTPUT_PATH):
    """
    Run a 2-D ray-tracing simulation of a spectrometer and save the result.

    Parameters
    ----------
    f1, f2            : float  focal lengths in mm
    sensor_width      : float  sensor width in mm
    start_wavelength  : float  shortest wavelength in nm
    end_wavelength    : float  longest wavelength in nm
    incident_angle    : float  grating incidence angle in degrees
    difracted_angle   : float  central diffracted angle in degrees (transmissive only)
    N                 : int    grooves per mm
    reflective        : bool   True → reflective (Czerny-Turner style),
                               False → transmissive (4f style)
    lens_aperture     : float  lens clear aperture in mm
    grating_aperture  : float  grating clear aperture in mm
    output_path       : str    file path for the saved PNG
    """
    wavelengths = list(np.linspace(start_wavelength, end_wavelength,
                                   N_WAVELENGTH_SAMPLES) * 1e-9)

    ymin, ymax = 0.1, -0.1
    scene = np.zeros((2, N_SOURCE_POINTS))
    scene[1, :] = np.linspace(ymin, ymax, N_SOURCE_POINTS)

    components = []

    # Collimating lens (F1) — same for both modes
    components.append(Lens(f=f1, aperture=lens_aperture,
                           pos=[f1, 0], theta=0, name="F1"))

    # Grating — transmissive flag controls the _get_angle physics
    components.append(Grating(ngroves=N, aperture=grating_aperture,
                               pos=[2 * f1, 0],
                               theta=np.radians(incident_angle),
                               transmissive=not reflective))

    # ── Layout: F2 and Sensor ─────────────────────────────────────────────────
    if reflective:
        # Reflected ray travels at theta_design (UP-LEFT for typical CT geometry).
        # place_dir: direction from grating toward F2/Sensor.
        # theta_elem = -place_dir so local incidence angle ≈ 0 (near-normal).
        theta_design = _reflective_theta_design(
            incident_angle, N, start_wavelength, end_wavelength)
        place_dir = theta_design
    else:
        # Transmitted ray travels at approximately -theta_design (slightly below horizontal).
        # theta_elem = theta_design gives local incidence ≈ 0.
        theta_design = np.radians(incident_angle + difracted_angle)
        place_dir = -theta_design

    theta_elem = -place_dir  # elements face toward incoming rays in both modes

    x1 = 2 * f1 + f2 * np.cos(place_dir)
    y1 = f2 * np.sin(place_dir)

    components.append(Lens(f=f2, aperture=grating_aperture,
                           pos=[x1, y1], theta=theta_elem, name="F2"))

    x2 = x1 + f2 * np.cos(place_dir)
    y2 = y1 + f2 * np.sin(place_dir)

    components.append(Sensor(aperture=sensor_width,
                              pos=[x2, y2], theta=theta_elem))

    rays, _, _ = initial_rays(scene, components[0], N_RAYS)
    colors = get_colors(len(wavelengths), N_RAYS * N_SOURCE_POINTS, cmap="rainbow")

    # Canvas bounds adapt to the layout direction
    all_x = [f1, 2 * f1, x1, x2]
    all_y = [0, y1, y2]
    x_margin = grating_aperture * 2
    y_margin = max(f1, abs(y2)) * 1.5
    canvas = Canvas(
        xlim=[min(all_x) - x_margin, max(all_x) + x_margin],
        ylim=[-y_margin, y_margin],
    )
    canvas.draw_components(components)

    for idx, lmb in enumerate(wavelengths):
        bundles = propagate_rays(components, rays, lmb=lmb)
        canvas.draw_rays(bundles, colors[idx], linewidth=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)
    canvas.close()
