#!/usr/bin/env python

import numpy as np
import scipy.linalg as lin


def angle_wrap(angle):
    """Wrap angle(s) to [-π, π]. Accepts scalars or arrays."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def propagate_rays(components, rays, lmb=525e-9):
    """
    Propagate rays through a list of optical components.

    Parameters
    ----------
    components : list of OpticalObject
    rays : (N, 3) array or list of 3-element lists  [x, y, angle]
    lmb : float  wavelength in metres

    Returns
    -------
    ray_bundles : ndarray, shape (3, N_rays, N_components+1)
    """
    rays = np.asarray(rays, dtype=float)          # (N_rays, 3)
    nrays = rays.shape[0]
    ncomp = len(components)

    bundles = np.full((3, nrays, ncomp + 1), np.nan)
    bundles[:, :, 0] = rays.T                     # (3, N_rays)

    for c_idx, component in enumerate(components):
        col_in = bundles[:, :, c_idx]             # (3, N_rays)
        col_out = np.full((3, nrays), np.nan)
        for r_idx in range(nrays):
            col_out[:, r_idx] = component.propagate(col_in[:, r_idx], lmb).reshape(3)
        bundles[:, :, c_idx + 1] = col_out

    return bundles


class OpticalObject:
    """Base class for all optical elements."""

    def __init__(self, aperture, pos, theta, name=None):
        self.theta = theta
        self.pos = pos
        self.aperture = aperture
        self.name = name
        self.H = self._create_xform()
        self.Hinv = lin.inv(self.H)

    def _create_xform(self):
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t,  cos_t, 0],
                      [0,      0,     1]], dtype=float)
        T = np.array([[1, 0, -self.pos[0]],
                      [0, 1, -self.pos[1]],
                      [0, 0,  1          ]], dtype=float)
        return R @ T

    def get_intersection(self, orig, theta):
        if np.isnan(theta):
            return np.full(3, np.nan)

        p = np.array([[orig[0]], [orig[1]], [1]], dtype=float)
        p_new = self.H @ p
        theta_new = theta + self.theta

        x_tf = 0.0
        y_tf = float(p_new[1, 0] - p_new[0, 0] * np.tan(theta_new))

        flag = np.nan if abs(y_tf) > self.aperture / 2.0 else 1.0

        p_tf = np.array([[x_tf], [y_tf], [1.0]])
        p_final = self.Hinv @ p_tf
        return np.array([float(p_final[0, 0]), float(p_final[1, 0]), flag])

    def propagate(self, point, lmb=None):
        dest = self.get_intersection(point[:2], point[2])
        if np.isnan(dest[2]):
            return dest
        dest[2] = self._get_angle(point, lmb, dest)
        return dest

    def _get_angle(self, point, lmb, dest):
        raise NotImplementedError


class FocalElement(OpticalObject):
    """
    Unified thin lens / spherical mirror element.

    Parameters
    ----------
    f : float  focal length (positive = converging)
    reflective : bool  True for spherical mirror, False (default) for lens
    """

    def __init__(self, f, aperture, pos, theta, reflective=False, name=""):
        self.f = f
        self.reflective = reflective
        self.type = "spherical_mirror" if reflective else "lens"
        if name == "":
            name = f"f = {f:g}"
        super().__init__(aperture, pos, theta, name)

    def _get_angle(self, point, lmb, dest):
        sign = -1.0 if self.reflective else 1.0
        focal_dest = self.Hinv @ np.array([[sign * self.f],
                                            [self.f * np.tan(point[2] + self.theta)],
                                            [1.0]])
        theta = np.arctan2(
            float(focal_dest[1, 0]) - dest[1],
            float(focal_dest[0, 0]) - dest[0],
        )
        if self.f < 0:
            theta += np.pi
        return float(theta)


# Convenience aliases
def Lens(f, aperture, pos, theta, name=""):
    return FocalElement(f=f, aperture=aperture, pos=pos, theta=theta,
                        reflective=False, name=name)


def SphericalMirror(f, aperture, pos, theta, name=""):
    return FocalElement(f=f, aperture=aperture, pos=pos, theta=theta,
                        reflective=True, name=name)


class Sensor(OpticalObject):
    """Sensor plane — terminates all rays."""

    def __init__(self, aperture, pos, theta, name="Sensor"):
        super().__init__(aperture, pos, theta, name)
        self.type = "sensor"

    def _get_angle(self, point, lmb, dest):
        return float("nan")


class Grating(OpticalObject):
    """Reflective or transmissive blazed diffraction grating."""

    def __init__(self, ngroves, aperture, pos, theta, m=1, transmissive=False,
                 name="Grating"):
        super().__init__(aperture, pos, theta, name)
        self.ngroves = ngroves
        self.m = m
        self.transmissive = transmissive
        self.type = "grating"

    def _get_angle(self, point, lmb, dest):
        incident_theta = point[2] + self.theta
        d = 1e-3 / self.ngroves
        sin_ref = np.sin(incident_theta) - self.m * lmb / d
        sin_ref = np.clip(sin_ref, -1.0, 1.0)
        refracted_theta = np.arcsin(sin_ref)
        if self.transmissive:
            # Beam continues forward through the grating
            return float(angle_wrap(refracted_theta - self.theta))
        else:
            # Reflective: beam bounces back (pi offset) with diffraction applied
            return float(angle_wrap(np.pi - refracted_theta - self.theta))


class Mirror(OpticalObject):
    """Flat mirror."""

    def __init__(self, aperture, pos, theta, name="Mirror"):
        super().__init__(aperture, pos, theta, name)
        self.type = "mirror"

    def _get_angle(self, point, lmb, dest):
        return float(angle_wrap(np.pi - point[2] - 2 * self.theta))


class DMD(OpticalObject):
    """Digital micromirror device with a deflection angle."""

    def __init__(self, deflection, aperture, pos, theta, name="DMD"):
        super().__init__(aperture, pos, theta, name)
        self.deflection = deflection
        self.type = "dmd"

    def _get_angle(self, point, lmb=None, dest=None):
        return float(angle_wrap(np.pi - point[2] - 2 * (self.theta + self.deflection)))


class Aperture(OpticalObject):
    """Hard aperture stop — passes rays, blocks those outside."""

    def __init__(self, aperture, pos, theta, name="Aperture"):
        super().__init__(aperture, pos, theta, name)
        self.type = "aperture"

    def _get_angle(self, point, lmb=None, dest=None):
        return float(point[2])
