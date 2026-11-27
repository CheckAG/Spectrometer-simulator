"""
Reference outputs from the Python core (spectrometer/) for the TypeScript port.

Run from the repo root:  python web/scripts/make_fixtures.py
Writes web/src/physics/fixtures.json; the Vitest suite checks the TS physics against it.
NaN is written as null (JSON has no NaN).
"""

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spectrometer.layout import trace, sensor_hits  # noqa: E402
from spectrometer.optics import (  # noqa: E402
    HG_AR_LINES, Spec, analyze, detector_signal, line_dip, recommend_groove_density,
)

BASE = dict(wl_min=400, wl_max=700, resolution=1.0, pixel_um=5.0, n_pixels=2048, f1=50, f2=50)
CASES = [  # (name, spec overrides, G or None = recommended)
    ("defaults", {}, None),
    ("reflective", dict(transmissive=False), None),
    ("reflective exact", dict(transmissive=False, exact_mirrors=True), None),
    ("reflective exact, loose target", dict(transmissive=False, exact_mirrors=True, resolution=2.0), 300),
    ("overflow 4f", dict(f1=150, f2=150), 300),
    ("overflow CT", dict(f1=150, f2=150, transmissive=False), 300),
    ("too few pixels", dict(resolution=0.1, n_pixels=512), 600),
    ("target missed", dict(resolution=0.3), 300),
    ("order overlap", dict(wl_min=400, wl_max=900), 300),
    ("mirrors collide", dict(transmissive=False, included_deg=10), 300),
    ("off-span blaze", dict(blaze_nm=300), 300),
    ("grazing", dict(transmissive=False, included_deg=77), 2400),
    ("manual angle 4f", dict(included_deg=25, beam_mm=2.0), 600),
    ("NIR CT", dict(wl_min=900, wl_max=1700, resolution=5, pixel_um=25, n_pixels=512,
                    f1=100, f2=100, transmissive=False), None),
]


def clean(x):
    """numpy → plain JSON values; NaN → None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def case(name, overrides, G):
    spec = Spec(**{**BASE, **overrides})
    recommended = recommend_groove_density(spec).G
    G = G or recommended
    d = analyze(spec, G)
    lines = [l for l in HG_AR_LINES if spec.wl_min <= l <= spec.wl_max] or [spec.wl_center]
    wl_pix, signal = detector_signal(spec, d, lines)
    wls = list(np.linspace(spec.wl_min, spec.wl_max, 5))
    traced = {}
    if d.ok:
        for exact in ([False, True] if not spec.transmissive else [False]):
            comps, bundles = trace(spec, d, wls, exact=exact)
            hits = sensor_hits(comps, bundles).reshape(len(wls), -1)
            traced["exact" if exact else "ideal"] = {
                "mean": [float(np.nanmean(h)) if np.isfinite(h).any() else None for h in hits],
                "sensor_pos": list(comps[-1].pos), "sensor_theta": float(comps[-1].theta),
            }
    design = {k: v for k, v in asdict(d).items() if k != "warnings"}
    return {
        "name": name, "spec": asdict(spec), "G": G, "recommended": recommended,
        "design": design,
        "warnings": [{"text": str(w), "field": w.field, "fixes": w.fixes} for w in d.warnings],
        "detector": {"lines": lines, "step": 64, "wl_pix": list(wl_pix[::64]), "signal": list(signal[::64]),
                     "dip_577": line_dip(wl_pix, signal, 576.96, 579.07)},
        "trace": {"wavelengths": wls, **traced},
    }


out = ROOT / "web" / "src" / "physics" / "fixtures.json"
out.write_text(json.dumps(clean([case(*c) for c in CASES]), indent=1, ensure_ascii=False, allow_nan=False))
print(f"wrote {len(CASES)} cases to {out.relative_to(ROOT)}")
