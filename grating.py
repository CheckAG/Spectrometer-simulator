#!/usr/bin/env python3

import os
import numpy as np

from visualize import Canvas, get_colors
from ray_utilities import initial_rays
from raytracing_v import Lens, Grating, Sensor, propagate_rays
from config import (N_WAVELENGTH_SAMPLES, N_RAYS, N_SOURCE_POINTS,
                    LENS_APERTURE, GRATING_APERTURE, SIMULATION_OUTPUT_PATH)


def design_rendering(f1, f2, sensor_width, start_wavelength, end_wavelength,
                     incident_angle, difracted_angle, N,
                     lens_aperture=LENS_APERTURE,
                     grating_aperture=GRATING_APERTURE,
                     output_path=SIMULATION_OUTPUT_PATH):
    """
    Run a 2-D ray-tracing simulation of a 4f spectrometer and save the result.

    Parameters
    ----------
    f1, f2            : float  focal lengths in mm
    sensor_width      : float  sensor width in mm
    start_wavelength  : float  shortest wavelength in nm
    end_wavelength    : float  longest wavelength in nm
    incident_angle    : float  grating incidence angle in degrees
    difracted_angle   : float  central diffracted angle in degrees
    N                 : int    grooves per mm
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

    components.append(Lens(f=f1, aperture=lens_aperture,
                           pos=[f1, 0], theta=0, name="F1"))

    components.append(Grating(ngroves=N, aperture=grating_aperture,
                               pos=[2 * f1, 0],
                               theta=np.radians(incident_angle),
                               transmissive=False))

    theta_design = np.radians(incident_angle + difracted_angle)
    x1 = 2 * f1 + f2 * np.cos(-theta_design)
    y1 = f2 * np.sin(-theta_design)

    components.append(Lens(f=f2, aperture=grating_aperture,
                           pos=[x1, y1], theta=theta_design, name="F2"))

    x2 = x1 + f2 * np.cos(-theta_design)
    y2 = y1 + f2 * np.sin(-theta_design)

    components.append(Sensor(aperture=sensor_width,
                              pos=[x2, y2], theta=theta_design))

    rays, _, _ = initial_rays(scene, components[0], N_RAYS)
    colors = get_colors(len(wavelengths), N_RAYS * N_SOURCE_POINTS, cmap="rainbow")

    canvas = Canvas([-5, 5 * grating_aperture], [-f1 * 2, f1 * 2])
    canvas.draw_components(components)

    for idx, lmb in enumerate(wavelengths):
        bundles = propagate_rays(components, rays, lmb=lmb)
        canvas.draw_rays(bundles, colors[idx], linewidth=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)
    canvas.close()
