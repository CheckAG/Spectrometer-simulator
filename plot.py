"""Interactive (Plotly) drawing of a traced design."""

import numpy as np
import plotly.graph_objects as go

from config import N_WAVELENGTH_SAMPLES, EXTRA_ORDERS
from grating import build_components, trace
from optics import Spec, Design

ORDER_STYLE = {  # order: (dash, opacity, legend text)
    1: ("solid", 1.0, "design order"),
    0: ("dot", 0.35, "undiffracted (stray light)"),
    2: ("dash", 0.35, "second order (overlap)"),
}
ELEMENT_COLOR = {"lens": "#4C8DFF", "spherical_mirror": "#2E9E5B", "grating": "#C04BD6",
                 "sensor": "#00A3B4", "mirror": "#555", "dmd": "#555", "aperture": "#555"}


def wavelength_rgb(nm):
    """Approximate perceived colour of a wavelength (Bruton); dim violet/red outside 380–780 nm."""
    if nm < 380:
        return "rgb(110,40,140)"
    if nm > 780:
        return "rgb(140,30,30)"
    for lo, hi, rgb in [(380, 440, lambda t: (1 - t, 0, 1)), (440, 490, lambda t: (0, t, 1)),
                        (490, 510, lambda t: (0, 1, 1 - t)), (510, 580, lambda t: (t, 1, 0)),
                        (580, 645, lambda t: (1, 1 - t, 0)), (645, 781, lambda t: (1, 0, 0))]:
        if nm < hi:
            r, g, b = rgb((nm - lo) / (hi - lo))
            break
    # Dim toward the edges of vision, but keep lines visible.
    fade = 0.4 + 0.6 * min(1, (nm - 380) / 40, (780 - nm) / 80)
    return f"rgb({r * fade * 255:.0f},{g * fade * 255:.0f},{b * fade * 255:.0f})"


def _paths(bundles, reach, start=0):
    """
    x, y arrays (None-separated) for every ray from point `start`: up to the
    first NaN point, plus the last segment extended by `reach` if the ray is
    still travelling.
    """
    xs, ys = [], []
    x, y, a = bundles
    for i in range(x.shape[0]):
        n = int(np.argmax(np.isnan(x[i]))) if np.isnan(x[i]).any() else x.shape[1]
        px, py = list(x[i, start:n]), list(y[i, start:n])
        if n == x.shape[1] and not np.isnan(a[i, -1]):
            px.append(x[i, -1] + reach * np.cos(a[i, -1]))
            py.append(y[i, -1] + reach * np.sin(a[i, -1]))
        xs += px + [None]
        ys += py + [None]
    return xs, ys


def _landing(bundles):
    """Mean point where an order meets the element after the grating, or None."""
    hit = bundles[:2, :, 3]
    if np.isnan(hit).all():
        ok = ~np.isnan(bundles[2, :, 2])
        if not ok.any():
            return None
        x, y, a = bundles[0, ok, 2], bundles[1, ok, 2], bundles[2, ok, 2]
        hit = np.vstack([x + 30 * np.cos(a), y + 30 * np.sin(a)])
    return np.nanmean(hit, axis=1)


def _glyph(c, back, transmissive):
    """
    Outline of an element in its local frame (x = along the normal, y = along
    the aperture); `back` (±1) is the side light leaves through for
    transmissive parts and the non-reflecting side for reflectors.
    Returns (x, y) of a closed polygon, schematic but proportioned to the aperture.
    """
    h = c.aperture / 2
    t = np.linspace(-h, h, 41)
    if c.type == "lens":                           # biconvex
        w = 0.06 * c.aperture * (1 - (t / h) ** 2) + 0.4
        return np.r_[w, -w[::-1], w[:1]], np.r_[t, t[::-1], t[:1]]
    if c.type == "spherical_mirror":               # concave face, flat back
        sag = (t ** 2 - h ** 2) / (4 * abs(c.f))  # edges reach toward the light
        x_face, x_back = -back * sag, np.full_like(t, back * 2.0)
        return np.r_[x_face, x_back[::-1], x_face[:1]], np.r_[t, t[::-1], t[:1]]
    if c.type == "grating":                        # substrate with sawtooth grooves
        n = 16
        ty = np.linspace(-h, h, 2 * n + 1)
        tooth = np.where(np.arange(2 * n + 1) % 2, 1.2, 0.0)
        face = -back if not transmissive else back  # grooves on the working face
        sub = -face * 3.0
        x = np.r_[face * tooth, sub, sub, 0]
        y = np.r_[ty, h, -h, -h]
        return x, y
    depth = 1.5 if c.type == "sensor" else 1.0     # sensor package, plane mirror, ...
    return np.array([0, 0, back * depth, back * depth, 0]), np.array([-h, h, h, -h, -h])


def _rgba(hex_color, a):
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) if len(hex_color) == 7 \
        else (int(ch * 2, 16) for ch in hex_color.lstrip("#"))
    return f"rgba({r},{g},{b},{a})"


def ray_figure(spec: Spec, design: Design, extra_orders=EXTRA_ORDERS, height=520):
    components, slit = build_components(spec, design)
    wavelengths = np.linspace(spec.wl_min, spec.wl_max, N_WAVELENGTH_SAMPLES)
    pts = np.array([c.pos for c in components] + [slit.mean(axis=1)])
    reach = 1.5 * np.ptp(pts, axis=0).max() + spec.grating_mm
    fig = go.Figure()

    # Before the grating every wavelength shares one path: draw it once, as white light.
    _, white = trace(spec, design, wavelengths[:1])
    xs, ys = _paths(white[:, :, :3], 0)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#9a9a9a", width=1),
                             showlegend=False, hovertemplate="white light<extra></extra>"))

    orders = [spec.m] + [m for m in extra_orders if m != spec.m]
    for m in reversed(orders):                     # design order drawn on top
        dash, opacity, text = ORDER_STYLE.get(m, ("dashdot", 0.45, ""))
        start = 2                                  # colour starts at the grating
        _, bundles = trace(spec, design, wavelengths, m=m)
        per_wl = bundles.shape[1] // len(wavelengths)
        exists = not np.isnan(bundles[2, :, 2]).all()
        label = f"m = {m}  {text}" + ("" if exists else " — not possible here")
        for k, wl in enumerate(wavelengths):
            xs, ys = _paths(bundles[:, k * per_wl:(k + 1) * per_wl], reach, start)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", opacity=opacity,
                line=dict(color=wavelength_rgb(wl), width=1 if m == spec.m else 0.8, dash=dash),
                name=label, legendgroup=f"m{m}", showlegend=False,
                hovertemplate=f"{wl:.0f} nm · order {m}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label,
                                 legendgroup=f"m{m}", legendrank=orders.index(m),
                                 line=dict(color="#9a9a9a", width=2, dash=dash)))
        spot = _landing(bundles)
        if spot is not None:
            fig.add_annotation(x=spot[0], y=spot[1], text=f"m = {m}", showarrow=True,
                               ax=30, ay=-30 + 22 * orders.index(m), arrowhead=2,
                               arrowsize=0.8, arrowcolor="#9a9a9a", font=dict(size=11),
                               bgcolor="rgba(128,128,128,0.25)", borderpad=2)

    centre = pts.mean(axis=0)
    for k, c in enumerate(components):
        h = c.aperture / 2
        color = ELEMENT_COLOR.get(c.type, "#555555")
        # Direction of the light arriving at this element (from the white-light trace).
        a = np.arctan2(np.nanmean(np.sin(white[2, :, k])), np.nanmean(np.cos(white[2, :, k])))
        nx, ny = np.subtract(c.to_global(np.ones(1), np.zeros(1)), c.to_global(np.zeros(1), np.zeros(1)))
        back = 1.0 if np.isnan(a) or nx[0] * np.cos(a) + ny[0] * np.sin(a) >= 0 else -1.0
        gx, gy = _glyph(c, back, spec.transmissive)
        ex, ey = c.to_global(gx, gy)
        detail = f"f = {abs(c.f):.0f} mm · " if hasattr(c, "f") else ""
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines", fill="toself", fillcolor=_rgba(color, 0.35),
            line=dict(color=color, width=1.5), showlegend=False,
            hovertemplate=f"{c.name} · {detail}aperture {c.aperture:.1f} mm<extra></extra>"))
        # Reflectors and the sensor: label on the dark side. Light passes through
        # transmissive parts, so label those past the end farther from the centre.
        if c.type == "lens" or (c.type == "grating" and spec.transmissive) or np.isnan(a):
            lx, ly = c.to_global(np.zeros(2), np.array([-h - 7, h + 7]))
        else:
            lx, ly = np.array([c.pos[0] + 9 * np.cos(a)]), np.array([c.pos[1] + 9 * np.sin(a)])
        far = int(np.argmax(np.hypot(lx - centre[0], ly - centre[1])))
        fig.add_annotation(x=lx[far], y=ly[far], text=f"<b>{c.name}</b>", showarrow=False,
                           font=dict(size=12, color=color))
    # Slit: two jaws across the beam; the gap is µm, so it is drawn to scale only by hover.
    sx, sy = slit.mean(axis=1)
    across = slit[:, 0] - slit[:, -1]
    across = across / np.hypot(*across) if np.hypot(*across) else np.array([0.0, 1.0])
    g, jaw = design.slit_um * 1e-3 / 2 + 0.3, 4.0
    jx = [sx + g * across[0], sx + (g + jaw) * across[0], None, sx - g * across[0], sx - (g + jaw) * across[0]]
    jy = [sy + g * across[1], sy + (g + jaw) * across[1], None, sy - g * across[1], sy - (g + jaw) * across[1]]
    fig.add_trace(go.Scatter(x=jx, y=jy, mode="lines", line=dict(color="#d0a030", width=4),
                             showlegend=False,
                             hovertemplate=f"Slit · {design.slit_um:.1f} µm<extra></extra>"))
    fig.add_annotation(x=sx - (g + jaw + 4) * across[0], y=sy - (g + jaw + 4) * across[1],
                       text="<b>Slit</b>", showarrow=False, font=dict(size=12, color="#d0a030"))

    pad = spec.grating_mm * 0.6
    fig.update_xaxes(range=[pts[:, 0].min() - pad, pts[:, 0].max() + pad], title="x (mm)",
                     zeroline=False)
    fig.update_yaxes(range=[pts[:, 1].min() - pad, pts[:, 1].max() + pad], title="y (mm)",
                     scaleanchor="x", scaleratio=1, zeroline=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      hoverlabel=dict(namelength=-1), dragmode="pan")
    return fig


def detector_figure(wl_pix, signal, lines_nm, span=None, height=360):
    """Simulated detector read-out vs wavelength; true lines dotted, design span shaded."""
    fig = go.Figure()
    if span:
        fig.add_vrect(x0=span[0], x1=span[1], fillcolor="#9a9a9a", opacity=0.08, line_width=0,
                      annotation_text="design span", annotation_position="top left")
    for line in lines_nm:
        fig.add_vline(x=line, line=dict(color=wavelength_rgb(line), width=1, dash="dot"),
                      opacity=0.7)
    pix = np.arange(len(signal))
    fig.add_trace(go.Scatter(
        x=wl_pix, y=signal, mode="lines", line=dict(color="#4C8DFF", width=1.5, shape="hvh"),
        customdata=pix, showlegend=False,
        hovertemplate="pixel %{customdata} · %{x:.2f} nm · %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="wavelength (nm)", range=[np.nanmin(wl_pix), np.nanmax(wl_pix)])
    fig.update_yaxes(title="signal (peak = 1)", rangemode="tozero")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), dragmode="zoom")
    return fig
