/** App inputs, their limits, and the URL query-string round trip (shareable links). */

import * as C from './physics/config'
import { type Spec, makeSpec } from './physics/optics'

export interface Inputs {
  transmissive: boolean
  wl_min: number
  wl_max: number
  resolution: number
  pixel_um: number
  n_pixels: number
  sampling: number
  f1: number
  f2: number
  beam_mm: number
  exact: boolean
  angle: number | null // null: follow the computed angle
  G: number | null // null: follow the recommendation
  blaze: number | null // null: computed blaze
}

export const DEFAULTS: Inputs = {
  transmissive: true,
  wl_min: C.DEFAULT_SPAN_START,
  wl_max: C.DEFAULT_SPAN_END,
  resolution: C.DEFAULT_SPECTRAL_RES,
  pixel_um: C.DEFAULT_PIXEL_SIZE,
  n_pixels: C.DEFAULT_N_PIXELS,
  sampling: C.DEFAULT_SAMPLING,
  f1: C.DEFAULT_F1,
  f2: C.DEFAULT_F2,
  beam_mm: C.LENS_APERTURE,
  exact: false,
  angle: null,
  G: null,
  blaze: null,
}

type NumKey = { [K in keyof Inputs]: Inputs[K] extends number | null ? K : never }[keyof Inputs]

export const LIMITS: Record<Exclude<NumKey, 'G'>, [number, number]> = {
  wl_min: [C.WAVELENGTH_MIN, C.WAVELENGTH_MAX],
  wl_max: [C.WAVELENGTH_MIN, C.WAVELENGTH_MAX],
  resolution: [C.SPECTRAL_RES_MIN, C.SPECTRAL_RES_MAX],
  pixel_um: [C.PIXEL_SIZE_MIN, C.PIXEL_SIZE_MAX],
  n_pixels: [C.N_PIXELS_MIN, C.N_PIXELS_MAX],
  sampling: [1, 10],
  f1: [C.FOCAL_LENGTH_MIN, C.FOCAL_LENGTH_MAX],
  f2: [C.FOCAL_LENGTH_MIN, C.FOCAL_LENGTH_MAX],
  beam_mm: [2, C.GRATING_APERTURE],
  angle: [C.INCLUDED_ANGLE_MIN, C.INCLUDED_ANGLE_MAX],
  blaze: [C.WAVELENGTH_MIN, C.WAVELENGTH_MAX],
}

export function clamp(key: keyof typeof LIMITS, v: number): number {
  const [lo, hi] = LIMITS[key]
  const c = Math.min(Math.max(v, lo), hi)
  return key === 'n_pixels' ? Math.round(c) : c
}

/** Inputs from a query string; anything missing or invalid falls back to the default. */
export function fromQuery(search: string): Inputs {
  const q = new URLSearchParams(search)
  const out: Inputs = { ...DEFAULTS }
  if (q.has('grating_type')) out.transmissive = q.get('grating_type') !== 'Reflective'
  if (q.has('exact')) out.exact = q.get('exact') === 'True'
  if (q.has('G') && C.GROOVE_OPTIONS.includes(Number(q.get('G')))) out.G = Number(q.get('G'))
  for (const key of Object.keys(LIMITS) as (keyof typeof LIMITS)[]) {
    const raw = q.get(key)
    const v = raw === null ? NaN : Number(raw)
    if (Number.isFinite(v)) (out as unknown as Record<string, number>)[key] = clamp(key, v)
  }
  if (out.wl_min >= out.wl_max) [out.wl_min, out.wl_max] = [DEFAULTS.wl_min, DEFAULTS.wl_max]
  return out
}

/** Query string for the current inputs; same keys as the Streamlit app, so links carry over. */
export function toQuery(s: Inputs): string {
  const q = new URLSearchParams()
  q.set('grating_type', s.transmissive ? 'Transmissive' : 'Reflective')
  for (const key of Object.keys(LIMITS) as (keyof typeof LIMITS)[]) {
    const v = s[key]
    if (v !== null) q.set(key, String(v))
  }
  if (s.G !== null) q.set('G', String(s.G))
  if (s.exact) q.set('exact', 'True')
  return q.toString()
}

export function toSpec(s: Inputs): Spec {
  return makeSpec({
    wl_min: s.wl_min,
    wl_max: s.wl_max,
    resolution: s.resolution,
    pixel_um: s.pixel_um,
    n_pixels: s.n_pixels,
    sampling: s.sampling,
    f1: s.f1,
    f2: s.f2,
    included_deg: s.angle,
    transmissive: s.transmissive,
    m: C.DIFFRACTION_ORDER,
    beam_mm: s.beam_mm,
    blaze_nm: s.blaze,
    exact_mirrors: s.exact,
  })
}

/** Apply a warning's fix (Spec overrides) to the inputs. */
export function applyFix(s: Inputs, fix: Partial<Spec>): Inputs {
  const out = { ...s }
  for (const [field, value] of Object.entries(fix)) {
    const key = ({ included_deg: 'angle', blaze_nm: 'blaze' } as Record<string, keyof typeof LIMITS>)[field] ?? field
    ;(out as unknown as Record<string, number | null>)[key] =
      value === null ? null : clamp(key as keyof typeof LIMITS, value as number)
  }
  return out
}
