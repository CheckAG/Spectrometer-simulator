#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines, cm


class Canvas:
    """2-D canvas for drawing optical components and ray paths."""

    def __init__(self, xlim, ylim, bbox=None, figsize=None, dpi=100):
        if figsize is not None:
            self._canvas, self.axes = plt.subplots(figsize=figsize)
        else:
            self._canvas, self.axes = plt.subplots()

        self.xlim = xlim
        self.ylim = ylim
        self.axes.axis("scaled")
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.grid(True)

        if bbox is None:
            bbox = {"facecolor": "yellow", "alpha": 0.5}
        self.bbox = bbox

    def draw_components(self, components):
        for component in components:
            xy = component.Hinv @ np.array([0, -component.aperture / 2, 1])
            ctype = getattr(component, "type", "")

            if ctype == "sensor":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.1,
                    height=component.aperture,
                    angle=-np.degrees(component.theta),
                    linestyle="dashed", hatch="+", color="c")
                artist.set_alpha(1)
            elif ctype == "lens":
                artist = patches.Ellipse(
                    xy=component.pos, width=component.aperture * 0.1,
                    height=component.aperture,
                    angle=-np.degrees(component.theta))
                artist.set_alpha(0.5)
            elif ctype == "spherical_mirror":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.05,
                    height=component.aperture,
                    angle=-np.degrees(component.theta), color="g")
                artist.set_alpha(1)
            elif ctype == "mirror":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.05,
                    height=component.aperture,
                    angle=-np.degrees(component.theta), color="k")
                artist.set_alpha(1)
            elif ctype == "grating":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.05,
                    height=component.aperture,
                    angle=-np.degrees(component.theta), hatch="/", color="m")
                artist.set_alpha(0.2)
            elif ctype == "dmd":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.1,
                    height=component.aperture,
                    angle=-np.degrees(component.theta),
                    linestyle="dashed", hatch="x", color="g")
                artist.set_alpha(1)
            elif ctype == "aperture":
                artist = patches.Rectangle(
                    xy=xy[:2], width=component.aperture * 0.02,
                    height=component.aperture,
                    angle=-np.degrees(component.theta), color="b")
                artist.set_alpha(0.5)
            else:
                raise ValueError(f"Unknown component type: '{ctype}'")

            self.axes.add_artist(artist)

            if component.name is not None:
                label_xy = component.Hinv @ np.array([8, -component.aperture / 2 - 8, 1])
                self.axes.text(label_xy[0], label_xy[1], component.name, bbox=self.bbox)

    def draw_rays(self, ray_bundles, colors=None, linewidth=0.5, membership=None):
        """
        Draw propagated rays.

        Parameters
        ----------
        ray_bundles : ndarray, shape (3, N_rays, N_components+1)
                      as returned by propagate_rays()
        colors : list of colours, one per ray
        linewidth : float
        membership : list of ints, one per ray (controls linestyle cycling)
        """
        nrays = ray_bundles.shape[1]

        if colors is None:
            colors = [np.random.rand(3) for _ in range(nrays)]

        if membership is None:
            membership = [0] * nrays

        if len(colors) != nrays:
            raise ValueError("len(colors) must equal number of rays")
        if len(membership) != nrays:
            raise ValueError("len(membership) must equal number of rays")

        linestyles = ["-", "--", ":"]

        for r_idx in range(nrays):
            ray = ray_bundles[:, r_idx, :]          # (3, N_comp+1)
            style = linestyles[membership[r_idx] % 3]
            color = colors[r_idx]
            nsteps = ray.shape[1]

            for idx in range(nsteps - 1):
                if np.isnan(ray[0, idx]):
                    break
                line = lines.Line2D(
                    [ray[0, idx], ray[0, idx + 1]],
                    [ray[1, idx], ray[1, idx + 1]],
                    color=color, linewidth=linewidth, linestyle=style)
                self.axes.add_line(line)

            # Extend the last valid ray segment to the canvas edge
            last = nsteps - 1
            if np.isnan(ray[0, last]):
                continue
            dist = np.hypot(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])
            xmax = ray[0, last] + dist * np.cos(ray[2, last])
            ymax = ray[1, last] + dist * np.sin(ray[2, last])
            line = lines.Line2D(
                [ray[0, last], xmax], [ray[1, last], ymax],
                color=color, linewidth=linewidth, linestyle=style)
            self.axes.add_line(line)

    def save(self, savename, dpi=150):
        self._canvas.savefig(savename, bbox_inches="tight", dpi=dpi)

    def close(self):
        plt.close(self._canvas)


def get_colors(nwvl, nrays, cmap="jet", flatten=False):
    """
    Return a list of RGBA colours for visualisation.

    Parameters
    ----------
    nwvl   : int  number of wavelengths
    nrays  : int  number of rays per wavelength
    cmap   : str  matplotlib colormap name
    flatten: bool if True, return a single flat list

    Returns
    -------
    colors_list : list of lists (or flat list if flatten=True)
    """
    colormap = plt.colormaps[cmap]
    colors = colormap(np.linspace(0, 1, nwvl))

    colors_list = []
    for idx in range(nwvl):
        row = [colors[idx, :3] for _ in range(nrays)]
        if flatten:
            colors_list += row
        else:
            colors_list.append(row)

    return colors_list
