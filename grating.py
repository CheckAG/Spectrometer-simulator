#!/usr/bin/env python3

import os, sys
# sys.path.append('../modules/')
import sys

import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from visualize import *
from ray_utilities import *
from raytracing_v import *
import streamlit as st


def design_rendering(f1, f2, lens_aperture, grating_aperture, N, sensor_width,
                     start_wavelength, end_wavelength, incident_angle, difracted_angle):
    # Create a spectrometer using a simple 4f system and diffraction grating
    f1 = f1              # Focal length of f1 lens
    f2 = f2              # Focal length of f2 lens
    lens_aperture = lens_aperture 
    grating_aperture = grating_aperture
    npoints = 2         # Number of light source points
    nrays = 20           # Number of light rays per point
    ymax = -0.1         # Limit of source plane. Controls spectral resolution
    ymin = 0.1
    ngroves = N       # Grove density of diffraction grating
    sensor_width = sensor_width

    # Simulate system for these wavelengths
    lmb = list(np.linspace(start_wavelength, end_wavelength, 10)*1e-9)

    components = []
    rays = []
    image_plane = -200
    # nrays = 20

    # Create three scene points
    scene = np.zeros((2, npoints))
    scene[1, :] = np.linspace(ymin, ymax, npoints)

    # Place a collimation lens
    components.append(Lens(f=f1,
                              aperture=lens_aperture,
                              pos=[f1, 0],
                              theta=0,
                              name='F1'))

    
    # Place a diffraction grating
    components.append(Grating(ngroves=ngroves,
                                 aperture=grating_aperture,
                                 pos=[2*f1, 0],
                                 theta=np.deg2rad(incident_angle),
                                 transmissive=False)
                                 )

    # Place a lens such that the central wavelength is centered on the sensor
    # theta_design = np.arcsin(lmb[len(lmb)//2]/(1e-3/ngroves))
    theta_design = np.deg2rad(incident_angle+difracted_angle)
    print(f"theta design : {np.rad2deg(theta_design)}")
    x1 = (2*f1) + f2*np.cos(-theta_design)
    y1 = f2*np.sin(-theta_design)

    # components.append(Aperture(aperture=lens_aperture,
    #                             pos=[x1,y1/2],
    #                             theta=theta_design))

    components.append(Lens(f=f2,
                              aperture=grating_aperture,
                              pos=[x1, y1],
                              theta=(theta_design),
                              name='F2'))

    # Place a sensor
    x2 = x1 + f2*np.cos(-theta_design)
    y2 = y1 + f2*np.sin(-theta_design)

    components.append(Sensor(aperture=sensor_width,
                                pos=[x2, y2],
                                theta=(theta_design)))

    # Get the initial rays
    [rays, ptdict, colors] = initial_rays(scene,
                                                        components[0],
                                                        nrays)
    # Create rainbow colors
    colors = get_colors(len(lmb), nrays*npoints, cmap='rainbow')
    
    # Create a new canvas
    canvas = Canvas([-5, 5*grating_aperture], [-f1*2, f1*2])

    # Draw the components
    canvas.draw_components(components)

    # Draw the rays for each wavelength
    for idx in range(len(lmb)):
        canvas.draw_rays(propagate_rays(components, rays,
                                           lmb=lmb[idx]), colors[idx],
                        linewidth=0.2)

    # Show the system
    # canvas.show()

    # Save a copy
    canvas.save('img/grating.png')
