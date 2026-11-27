#!/usr/bin/env python3

import numpy as np
from matplotlib import colormaps, patches
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# type → (patch width as fraction of aperture, style kwargs)
_STYLES = {
    "sensor":           (0.1,  dict(linestyle="dashed", hatch="+", color="c")),
    "spherical_mirror": (0.05, dict(color="g")),
    "mirror":           (0.05, dict(color="k")),
    "grating":          (0.05, dict(hatch="/", color="m", alpha=0.4)),
    "dmd":              (0.1,  dict(linestyle="dashed", hatch="x", color="g")),
    "aperture":         (0.02, dict(color="b", alpha=0.5)),
}


class Canvas:
    """2-D canvas for drawing optical components and ray paths.

    Uses a bare Figure (no pyplot), so it never opens a window or leaks figures.
    """

    def __init__(self, xlim, ylim, figsize=None, bbox=None):
        self._canvas = Figure(figsize=figsize)
        self.axes = self._canvas.subplots()
        self.xlim, self.ylim = xlim, ylim
        self.axes.set_aspect("equal")
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.grid(True)
        self.bbox = bbox or {"facecolor": "yellow", "alpha": 0.5}

    def draw_components(self, components):
        for c in components:
            angle = -np.degrees(c.theta)
            if c.type == "lens":
                artist = patches.Ellipse(xy=c.pos, width=c.aperture * 0.1,
                                         height=c.aperture, angle=angle, alpha=0.5)
            elif c.type in _STYLES:
                frac, style = _STYLES[c.type]
                xy = c.Hinv @ np.array([0, -c.aperture / 2, 1])
                artist = patches.Rectangle(xy=xy[:2], width=c.aperture * frac,
                                           height=c.aperture, angle=angle, **style)
            else:
                raise ValueError(f"Unknown component type: '{c.type}'")
            self.axes.add_artist(artist)

            if c.name is not None:
                label_xy = c.Hinv @ np.array([8, -c.aperture / 2 - 8, 1])
                self.axes.text(label_xy[0], label_xy[1], c.name, bbox=self.bbox)

    def draw_rays(self, ray_bundles, colors=None, linewidth=0.5, linestyle="-"):
        """
        Draw propagated rays.

        ray_bundles : ndarray (3, N_rays, N_components+1) from propagate_rays()
        colors      : one colour per ray
        Rays are drawn up to their first NaN point; rays still travelling after
        the last element are extended to the canvas edge.
        """
        nrays = ray_bundles.shape[1]
        if colors is None:
            colors = np.random.rand(nrays, 3)
        if len(colors) != nrays:
            raise ValueError("len(colors) must equal number of rays")

        x, y, ang = ray_bundles
        # Valid up to the first NaN, per ray.
        valid = np.cumprod(~np.isnan(x), axis=1).astype(bool)
        seg_ok = valid[:, :-1] & valid[:, 1:]
        segs = np.stack([np.stack([x[:, :-1], y[:, :-1]], -1),
                         np.stack([x[:, 1:], y[:, 1:]], -1)], axis=2)   # (N, S, 2, 2)

        # Extension of the last segment for rays still travelling.
        dist = np.hypot(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])
        last = np.stack([x[:, -1], y[:, -1]], -1)
        ext = np.stack([last, last + dist * np.stack([np.cos(ang[:, -1]), np.sin(ang[:, -1])], -1)], 1)
        ext_ok = valid[:, -1] & ~np.isnan(ang[:, -1])

        colors = np.asarray(colors, dtype=float)
        ray_idx = np.repeat(np.arange(nrays), seg_ok.shape[1]).reshape(seg_ok.shape)
        all_segs = np.concatenate([segs[seg_ok], ext[ext_ok]])
        all_cols = np.concatenate([colors[ray_idx[seg_ok]], colors[ext_ok]])
        self.axes.add_collection(LineCollection(all_segs, colors=all_cols,
                                                linewidths=linewidth, linestyles=linestyle))

    def save(self, savename, dpi=150):
        self._canvas.savefig(savename, bbox_inches="tight", dpi=dpi)

    def close(self):
        """Kept for compatibility; a bare Figure needs no closing."""


def get_colors(nwvl, nrays, cmap="jet", flatten=False):
    """
    RGB colours: one row of `nrays` identical colours per wavelength.
    Returns a list of lists (or one flat list if flatten=True).
    """
    rgb = colormaps[cmap](np.linspace(0, 1, nwvl))[:, :3]
    rows = [[rgb[i]] * nrays for i in range(nwvl)]
    return [c for row in rows for c in row] if flatten else rows
