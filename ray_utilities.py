#!/usr/bin/env python3

import numpy as np


def initial_rays(scene, objective, nrays=10):
    """
    Generate a fan of rays from each scene point directed toward the objective lens.

    Parameters
    ----------
    scene     : (2, N) array of source point positions
    objective : OpticalObject  the first optical element (sets the angular range)
    nrays     : int  rays per scene point

    Returns
    -------
    rays      : list of [x, y, angle] for every ray
    ptdict    : list of index ranges, one per scene point
    colors    : list of random RGBA tuples (one per ray)
    """
    rays = []
    colors = []
    ptdict = []

    n_points = scene.shape[1]
    x0, y0 = objective.pos
    r = objective.aperture
    theta = np.pi / 2 - objective.theta

    x1 = x0 + 0.5 * r * np.cos(theta)
    y1 = y0 + 0.5 * r * np.sin(theta)
    x2 = x0 - 0.5 * r * np.cos(theta)
    y2 = y0 - 0.5 * r * np.sin(theta)

    for idx in range(n_points):
        theta_min = np.arctan2(y1 - scene[1, idx], x1 - scene[0, idx])
        theta_max = np.arctan2(y2 - scene[1, idx], x2 - scene[0, idx])
        rays += ray_fan(scene[:, idx], [theta_min, theta_max], nrays)
        colors += [tuple(np.random.rand(4))] * nrays
        ptdict.append(np.arange(idx * nrays, (idx + 1) * nrays))

    return rays, ptdict, colors


def ray_fan(pos, theta_lim, nrays):
    """
    Create a fan of rays from a single point.

    Returns
    -------
    rays : list of [x, y, angle]
    """
    angles = np.linspace(theta_lim[0], theta_lim[1], nrays)
    return [[pos[0], pos[1], a] for a in angles]


def throughput(ray_bundles):
    """
    Fraction of rays that make it through the full system.

    Parameters
    ----------
    ray_bundles : ndarray, shape (3, N_rays, N_components+1)
    """
    last_x = ray_bundles[0, :, -1]           # (N_rays,)
    return float(np.sum(~np.isnan(last_x)) / last_x.size)


def vignetting(ray_bundles, ptdict):
    """
    Per-scene-point throughput estimate.

    Parameters
    ----------
    ray_bundles : ndarray, shape (3, N_rays, N_components+1)
    ptdict      : list of index arrays (one per scene point)

    Returns
    -------
    vign : ndarray of per-point throughput fractions
    """
    propagated = ~np.isnan(ray_bundles[0, :, -1])    # (N_rays,)
    vign = np.array([propagated[idx].mean() for idx in ptdict])
    return vign
