#!/usr/bin/env python
"""
Sequential 2-D ray tracer with ideal (thin) elements.

A ray is [x, y, angle]. Every element works on arrays of rays, shape (3, N).
Each element is a line segment in its local frame (x = 0, |y| ≤ aperture/2);
`theta` rotates global → local (local angle = global angle + theta).

Note: grating angles here use the opposite sign for β to optics.py
(sin β_tracer = sin α − mλ/d); see optics.py docstring.
"""

import numpy as np


def angle_wrap(angle):
    """Wrap angle(s) to [-π, π]. Accepts scalars or arrays."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def propagate_rays(components, rays, lmb=525e-9):
    """
    Propagate rays through a list of optical components.

    Parameters
    ----------
    components : list of OpticalObject
    rays : (N, 3) array or list of [x, y, angle]
    lmb : float or (N,) array, wavelength in metres (one per ray allowed)

    Returns
    -------
    ray_bundles : ndarray, shape (3, N_rays, N_components+1)
    """
    rays = np.asarray(rays, dtype=float).reshape(-1, 3)
    bundles = np.full((3, rays.shape[0], len(components) + 1), np.nan)
    bundles[:, :, 0] = rays.T
    for c_idx, component in enumerate(components):
        bundles[:, :, c_idx + 1] = component.propagate(bundles[:, :, c_idx], lmb)
    return bundles


class OpticalObject:
    """Base class for all optical elements."""

    type = ""

    def __init__(self, aperture, pos, theta, name=None):
        self.theta = theta
        self.pos = pos
        self.aperture = aperture
        self.name = name
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=float)
        T = np.array([[1, 0, -pos[0]], [0, 1, -pos[1]], [0, 0, 1]], dtype=float)
        self.H = R @ T
        self.Hinv = np.linalg.inv(self.H)

    def to_global(self, x_local, y_local):
        p = self.Hinv @ np.vstack([x_local, y_local, np.ones_like(y_local)])
        return p[0], p[1]

    def get_intersection(self, orig, theta):
        """
        Hit points (global x, y) of rays on the element line, plus masks:
        `ahead` (ray actually travels to the element) and `inside` (within aperture).
        """
        p = self.H @ np.vstack([orig, np.ones(orig.shape[1])])
        th = theta + self.theta
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -p[0] / np.cos(th)
            y = p[1] + t * np.sin(th)
        ahead = np.isfinite(t) & (t > 1e-9)
        inside = ahead & (np.abs(y) <= self.aperture / 2 + 1e-9)  # edge rays aimed exactly at the rim
        x, y = self.to_global(np.zeros_like(y), y)
        return x, y, ahead, inside

    def propagate(self, rays, lmb=None):
        """rays: (3,) or (3, N). Same shape out. Blocked rays get angle NaN;
        rays that never reach the element get NaN everywhere."""
        rays = np.asarray(rays, dtype=float)
        r = rays.reshape(3, -1)
        x, y, ahead, inside = self.get_intersection(r[:2], r[2])
        with np.errstate(invalid="ignore"):
            angle = self._get_angle(r, lmb, np.vstack([x, y]))
        out = np.vstack([np.where(ahead, x, np.nan),
                         np.where(ahead, y, np.nan),
                         np.where(inside, angle, np.nan)])
        return out.reshape(rays.shape)

    def _get_angle(self, rays, lmb, dest):
        raise NotImplementedError


class FocalElement(OpticalObject):
    """
    Ideal thin lens / spherical mirror. A ray at local angle a is sent through
    the focal-plane point (±f, f·tan a), so rays from one point on the front
    focal plane leave parallel. No aberrations.
    """

    def __init__(self, f, aperture, pos, theta, reflective=False, name=""):
        self.f = f
        self.reflective = reflective
        self.type = "spherical_mirror" if reflective else "lens"
        super().__init__(aperture, pos, theta, name or f"f = {f:g}")

    def _get_angle(self, rays, lmb, dest):
        sign = -1.0 if self.reflective else 1.0
        local_a = rays[2] + self.theta
        fx, fy = self.to_global(np.full_like(local_a, sign * self.f), self.f * np.tan(local_a))
        theta = np.arctan2(fy - dest[1], fx - dest[0])
        return theta + np.pi if self.f < 0 else theta


def Lens(f, aperture, pos, theta, name=""):
    return FocalElement(f=f, aperture=aperture, pos=pos, theta=theta,
                        reflective=False, name=name)


def SphericalMirror(f, aperture, pos, theta, name=""):
    return FocalElement(f=f, aperture=aperture, pos=pos, theta=theta,
                        reflective=True, name=name)


class Sensor(OpticalObject):
    """Sensor plane: terminates all rays."""
    type = "sensor"

    def __init__(self, aperture, pos, theta, name="Sensor"):
        super().__init__(aperture, pos, theta, name)

    def _get_angle(self, rays, lmb, dest):
        return np.full(rays.shape[1], np.nan)


class Grating(OpticalObject):
    """Reflective or transmissive grating, order m. Evanescent orders → NaN."""
    type = "grating"

    def __init__(self, ngroves, aperture, pos, theta, m=1, transmissive=False,
                 name="Grating"):
        super().__init__(aperture, pos, theta, name)
        self.ngroves = ngroves
        self.m = m
        self.transmissive = transmissive

    def _get_angle(self, rays, lmb, dest):
        d = 1e-3 / self.ngroves
        beta = np.arcsin(np.sin(rays[2] + self.theta) - self.m * np.asarray(lmb) / d)
        if self.transmissive:
            return angle_wrap(beta - self.theta)
        return angle_wrap(np.pi - beta - self.theta)


class Mirror(OpticalObject):
    """Flat mirror."""
    type = "mirror"

    def __init__(self, aperture, pos, theta, name="Mirror"):
        super().__init__(aperture, pos, theta, name)

    def _get_angle(self, rays, lmb, dest):
        return angle_wrap(np.pi - rays[2] - 2 * self.theta)


class DMD(OpticalObject):
    """Digital micromirror device with a deflection angle."""
    type = "dmd"

    def __init__(self, deflection, aperture, pos, theta, name="DMD"):
        super().__init__(aperture, pos, theta, name)
        self.deflection = deflection

    def _get_angle(self, rays, lmb=None, dest=None):
        return angle_wrap(np.pi - rays[2] - 2 * (self.theta + self.deflection))


class Aperture(OpticalObject):
    """Hard aperture stop: passes rays inside, blocks those outside."""
    type = "aperture"

    def __init__(self, aperture, pos, theta, name="Aperture"):
        super().__init__(aperture, pos, theta, name)

    def _get_angle(self, rays, lmb=None, dest=None):
        return rays[2]
