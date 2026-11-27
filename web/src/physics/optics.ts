/**
 * Spectrometer design physics — port of spectrometer/optics.py (the reference).
 *
 * Grating equation (Palmer & Loewen): m·λ·G = sin α + sin β, both angles from
 * the grating normal, positive on the same side. The layout is fixed by the
 * included angle 2K = α − β: sin φ = mλG / (2 cos K), α = φ + K, β = φ − K.
 * Units: wavelength nm, lengths mm, pixels µm, angles degrees, G lines/mm.
 * Field names stay snake_case to match the Python core and its fixtures.
 */

import {
  AUTO_ANGLE_MARGIN,
  FOCAL_LENGTH_MAX,
  GRATING_APERTURE,
  GROOVE_OPTIONS,
  LENS_APERTURE,
} from './config'
import { f as fx, g, pct } from './format'
import { aberrationBlur } from './layout'

const NM_TO_MM = 1e-6
const UM_TO_MM = 1e-3
const rad = (d: number) => (d * Math.PI) / 180
const deg = (r: number) => (r * 180) / Math.PI

export interface Spec {
  wl_min: number
  wl_max: number
  resolution: number // nm, target FWHM
  pixel_um: number
  n_pixels: number
  f1: number
  f2: number
  included_deg: number | null // null → auto
  transmissive: boolean
  m: number
  sampling: number // pixels per resolution element
  beam_mm: number
  grating_mm: number
  n_glass: number
  blaze_nm: number | null // null → default
  exact_mirrors: boolean
}

export type SpecInput = Pick<Spec, 'wl_min' | 'wl_max' | 'resolution' | 'pixel_um' | 'n_pixels' | 'f1' | 'f2'> &
  Partial<Spec>

export function makeSpec(s: SpecInput): Spec {
  return {
    included_deg: null,
    transmissive: true,
    m: 1,
    sampling: 3.0,
    beam_mm: LENS_APERTURE,
    grating_mm: GRATING_APERTURE,
    n_glass: 1.5,
    blaze_nm: null,
    exact_mirrors: false,
    ...s,
  }
}

export const wlCenter = (s: Spec) => (s.wl_min + s.wl_max) / 2
export const sensorMm = (s: Spec) => s.n_pixels * s.pixel_um * UM_TO_MM

/** A warning: message, the input it is about, and one-click fixes {label: Spec overrides}. */
export interface Issue {
  text: string
  field: keyof Spec | null
  fixes: Record<string, Partial<Spec>>
}
const issue = (text: string, field: keyof Spec | null = null, fixes: Record<string, Partial<Spec>> = {}): Issue => ({
  text,
  field,
  fixes,
})

export interface Design {
  G: number
  included_deg: number
  alpha: number
  beta_center: number
  beta_min: number
  beta_max: number
  blaze_nm: number
  blaze_angle: number
  dispersion: number // rad/nm
  rld: number // nm/mm
  slit_um: number
  spectrum_mm: number
  fold1: number
  fold2: number
  resolution_image: number
  resolution_grating: number
  resolution: number // combined (includes aberrations when traced)
  warnings: Issue[]
  mirrors_clear: boolean
  resolution_aberration: number
  aberration_wl: number[]
  aberration_nm: number[]
}

export const designOk = (d: Design) => Number.isFinite(d.resolution)

const asinDeg = (x: number) => deg(Math.asin(x)) // NaN when |x| > 1

/** β (deg) for incidence α. */
export const diffractionAngle = (wl: number, G: number, alphaDeg: number, m = 1) =>
  asinDeg(m * wl * NM_TO_MM * G - Math.sin(rad(alphaDeg)))

/** (α, β) in deg that send wl along a layout with included angle α − β. */
export function gratingAngles(wl: number, G: number, includedDeg: number, m = 1): [number, number] {
  const K = rad(includedDeg) / 2
  const phi = asinDeg((m * wl * NM_TO_MM * G) / (2 * Math.cos(K)))
  return [phi + deg(K), phi - deg(K)]
}

/** dβ/dλ in rad/nm at fixed incidence. */
export const angularDispersion = (G: number, betaDeg: number, m = 1) => (m * G * NM_TO_MM) / Math.cos(rad(betaDeg))

/** Blaze wavelength centring the span in the 0.67–1.8 λB window (geometric centre). */
export const defaultBlazeWavelength = (wlMin: number, wlMax: number) => Math.sqrt((wlMax / 1.8) * (wlMin / 0.67))

const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))

/** Relative efficiency (peak 1) in order m, scalar theory: sinc²(m − λB/λ). */
export const blazeEfficiency = (wl: number, blazeNm: number, m = 1) => sinc(m - blazeNm / wl) ** 2

/** Facet angle (deg) that puts peak efficiency at wlBlaze. */
export function blazeAngle(wlBlaze: number, G: number, includedDeg: number, m = 1, transmissive = false, nGlass = 1.5) {
  const [alpha, beta] = gratingAngles(wlBlaze, G, includedDeg, m)
  if (!transmissive) return (alpha + beta) / 2
  const delta = rad(alpha + beta) // facet as a small prism bending the beam by δ
  return deg(Math.atan(Math.sin(delta) / (nGlass - Math.cos(delta))))
}

/** Czerny–Turner mirror off-axis angle (deg) to clear an element of half-size `clearance` at `distance`. */
export const foldAngle = (clearance: number, distance: number) =>
  Math.max(5.0, deg(0.5 * Math.asin(Math.min(clearance / distance, 0.9))))

/** Default included angle: 0 transmissive; reflective = smallest clearing angle + margin. */
export function autoIncludedAngle(spec: Spec, G: number, margin = AUTO_ANGLE_MARGIN): number {
  if (spec.transmissive) return 0
  const geometric = { ...spec, exact_mirrors: false } // clearance is geometric: skip tracing
  for (let angle = 0; angle <= 90; angle++) {
    const d = analyze({ ...geometric, included_deg: angle }, G, false)
    if (designOk(d) && d.mirrors_clear) return Math.min(angle + margin, 90)
  }
  return 90
}

/** First value from `start` (by `step`) at which the spectrum fits the sensor. */
function walkUntilFits(
  spec: Spec,
  G: number,
  overrides: (v: number) => Partial<Spec>,
  start: number,
  step: number,
  maxSteps = 400,
): number | null {
  let value = start
  for (let i = 0; i < maxSteps; i++) {
    if (value <= 0) return null
    const s = { ...spec, exact_mirrors: false, ...overrides(value) }
    const d = analyze(s, G, false)
    if (designOk(d) && d.spectrum_mm <= sensorMm(s)) return value
    value += step
  }
  return null
}

/** First of `values` at which the traced resolution meets the target (and fits, if asked). */
function walkUntilResolves(
  spec: Spec,
  G: number,
  overrides: (v: number) => Partial<Spec>,
  values: number[],
  fit = false,
): number | null {
  for (const v of values) {
    const s = { ...spec, ...overrides(v) }
    const d = analyze(s, G, false)
    if (designOk(d) && d.resolution <= s.resolution && (!fit || d.spectrum_mm <= sensorMm(s))) return v
  }
  return null
}

const arange = (start: number, stop: number, step: number) => {
  const out: number[] = []
  for (let v = start; step > 0 ? v < stop : v > stop; v += step) out.push(v)
  return out
}

/** Evaluate a spectrometer built from `spec` with a G lines/mm grating. */
export function analyze(specIn: Spec, G: number, fixes = true): Design {
  const asked = specIn
  const spec: Spec & { included_deg: number } = {
    ...specIn,
    included_deg: specIn.included_deg ?? autoIncludedAngle(specIn, G),
  }
  const { m, f1, f2 } = spec
  const center = wlCenter(spec)
  const sensor = sensorMm(spec)
  let mirrorsClear = true
  const warnings: Issue[] = []

  const [alpha, betaC] = gratingAngles(center, G, spec.included_deg, m)
  const betaMin = diffractionAngle(spec.wl_min, G, alpha, m)
  const betaMax = diffractionAngle(spec.wl_max, G, alpha, m)

  const blazeNm = spec.blaze_nm || defaultBlazeWavelength(spec.wl_min, spec.wl_max)
  const thetaB = blazeAngle(blazeNm, G, spec.included_deg, m, spec.transmissive, spec.n_glass)

  const fold1 = spec.transmissive ? 0 : foldAngle((spec.grating_mm + spec.beam_mm) / 2, f1)
  const fold2 = spec.transmissive ? 0 : foldAngle((spec.grating_mm + sensor) / 2, f2)

  // Detector on the focal plane, tilted by fold2: x(λ) = f2·[tan(Δβ − fold2) + tan(fold2)].
  const t2 = rad(fold2)
  const xAt = (b: number) => f2 * (Math.tan(rad(b - betaC) - t2) + Math.tan(t2))
  const spectrumMm = Math.abs(xAt(betaMax) - xAt(betaMin))
  const f2Eff = f2 / Math.cos(t2) ** 2

  const D = angularDispersion(G, betaC, m)
  const rld = 1 / (f2Eff * D)

  // Slit imaged with magnification (f2/f1)·(cos α / cos β).
  const imageMm = spec.sampling * spec.pixel_um * UM_TO_MM
  const mag = (f2Eff / f1) * (Math.cos(rad(alpha)) / Math.cos(rad(betaC)))
  const slitUm = imageMm / mag / UM_TO_MM

  let footprint = spec.beam_mm / Math.cos(rad(alpha))
  if (footprint > spec.grating_mm) {
    warnings.push(
      issue(
        `Beam footprint on grating (${fx(footprint, 1)} mm) exceeds grating size (${spec.grating_mm} mm); light is clipped.`,
        'included_deg',
      ),
    )
    footprint = spec.grating_mm
  }

  const resImage = rld * imageMm
  const resGrating = center / (m * G * footprint)
  let resolution = Math.hypot(resImage, resGrating)

  // |angle| ≥ 90° (or NaN): light would graze or come from behind.
  if (![alpha, betaMin, betaMax].every((a) => Math.abs(a) < 90)) {
    warnings.push(
      issue(
        `${G} lines/mm cannot diffract the full span at this included angle (an angle reaches 90°). Use fewer lines/mm or a smaller angle.`,
        'included_deg',
        { 'Use computed angle': { included_deg: null } },
      ),
    )
    resolution = NaN
  }

  const common = {
    G,
    included_deg: spec.included_deg,
    alpha,
    beta_center: betaC,
    beta_min: betaMin,
    beta_max: betaMax,
    blaze_nm: blazeNm,
    blaze_angle: thetaB,
    dispersion: D,
    rld,
    slit_um: slitUm,
    spectrum_mm: spectrumMm,
    fold1,
    fold2,
    resolution_image: resImage,
    resolution_grating: resGrating,
  }
  let aberration = { resolution_aberration: 0, aberration_wl: [] as number[], aberration_nm: [] as number[] }
  if (spec.exact_mirrors && !spec.transmissive && Number.isFinite(resolution)) {
    const probe: Design = { ...common, resolution, warnings: [], mirrors_clear: true, ...aberration }
    const [wl, blur] = aberrationBlur(spec, probe)
    const ideal = resolution
    const worst = blur.some(Number.isNaN) ? NaN : Math.max(...blur) // NaN: no traced light → unusable
    resolution = Math.hypot(ideal, worst)
    aberration = { resolution_aberration: worst, aberration_wl: wl, aberration_nm: blur }
    if (Number.isNaN(worst)) {
      warnings.push(
        issue(
          'With exact mirrors part of the beam is clipped before the sensor, so no resolution can be traced. Try fewer lines/mm or the computed angle.',
          'included_deg',
        ),
      )
    } else if (resolution > spec.resolution && ideal <= spec.resolution) {
      const remedies: Record<string, Partial<Spec>> = {}
      if (fixes) {
        // Equal mirrors keep the Czerny–Turner coma cancellation; prefer a length that also fits.
        const lengths = arange(Math.ceil(Math.max(f1, f2) / 10) * 10, FOCAL_LENGTH_MAX + 1, 10)
        const same = (v: number) => ({ f1: v, f2: v })
        const fl = walkUntilResolves(asked, G, same, lengths, true) ?? walkUntilResolves(asked, G, same, lengths)
        const beam = walkUntilResolves(asked, G, (v) => ({ beam_mm: v }), arange(Math.floor(spec.beam_mm) - 1, 2, -1))
        if (fl) remedies[`f1 = f2 = ${g(fl)} mm`] = same(fl)
        if (beam) remedies[`Beam ${g(beam)} mm`] = { beam_mm: beam }
      }
      warnings.push(
        issue(
          `Mirror aberrations add up to ${fx(worst, 2)} nm of blur, so the resolution is ${g(resolution, 3)} nm instead of ${g(ideal, 3)} nm. Fast, strongly folded mirrors cause it: longer equal focal lengths or a narrower beam reduce it.`,
          'f2',
          remedies,
        ),
      )
    }
  }

  const span = spec.wl_max - spec.wl_min
  if (spec.n_pixels < (spec.sampling * span) / spec.resolution) {
    const need = Math.ceil((spec.sampling * span) / spec.resolution)
    const best = Math.ceil(((spec.sampling * span) / spec.n_pixels) * 100) / 100
    warnings.push(
      issue(
        `Detector has ${spec.n_pixels} pixels but the span at the target resolution needs ≥ ${need} (at ${g(spec.sampling)} px per resolution element). No grating fixes this; widen the target or use a longer sensor.`,
        'n_pixels',
        { [`Target ${g(best)} nm`]: { resolution: best }, [`Use ${need} pixels`]: { n_pixels: need } },
      ),
    )
  }
  if (spectrumMm > sensor) {
    const fit = sensor / spectrumMm // first guesses scale linearly; the walks refine them
    const remedies: Record<string, Partial<Spec>> = {}
    if (fixes) {
      const f2Fit = walkUntilFits(spec, G, (v) => ({ f2: v }), Math.floor(f2 * fit), -1)
      const half = walkUntilFits(
        spec,
        G,
        (h) => ({ wl_min: Math.ceil(center - h), wl_max: Math.floor(center + h) }),
        Math.floor((span * fit) / 2),
        -1,
      )
      const px = walkUntilFits(spec, G, (n) => ({ n_pixels: n }), Math.ceil(spectrumMm / (spec.pixel_um * UM_TO_MM)), 16)
      if (f2Fit && f2Fit > 0) remedies[`f2 = ${g(f2Fit)} mm`] = { f2: f2Fit }
      if (half && half > 0) {
        const lo = Math.ceil(center - half)
        const hi = Math.floor(center + half)
        remedies[`Span ${g(lo)}–${g(hi)} nm`] = { wl_min: lo, wl_max: hi }
      }
      if (px) remedies[`Use ${px} pixels`] = { n_pixels: px }
    }
    const labels = Object.keys(remedies)
    const hint = labels.length
      ? labels.join(' or ')
      : `f2 ≤ ${fx(f2 * fit, 0)} mm, or a sensor ≥ ${fx(spectrumMm, 1)} mm, or a span of ~${fx(span * fit, 0)} nm`
    warnings.push(
      issue(
        `Spectrum is ${fx(spectrumMm, 1)} mm long but the sensor is only ${fx(sensor, 1)} mm; the ends of the span fall off. To fit: ${hint}.`,
        'f2',
        remedies,
      ),
    )
  }
  if (resolution > spec.resolution && !aberration.resolution_aberration) {
    const relaxed = Math.ceil(resolution * 100) / 100
    warnings.push(
      issue(`Achieved resolution ${g(resolution, 3)} nm is worse than the ${g(spec.resolution)} nm target.`, 'resolution', {
        [`Target ${g(relaxed)} nm`]: { resolution: relaxed },
      }),
    )
  }
  if (!spec.transmissive && Number.isFinite(alpha)) {
    // Mirrors sit f1 and f2 from the grating, (α − β) apart as seen from it.
    const gap = Math.sqrt(f1 ** 2 + f2 ** 2 - 2 * f1 * f2 * Math.cos(rad(spec.included_deg)))
    const m2Mm =
      ((spec.beam_mm * Math.cos(rad(betaC))) / Math.cos(rad(alpha)) + f2 * Math.abs(rad(betaMax - betaMin))) /
      Math.cos(t2)
    mirrorsClear = gap >= (spec.beam_mm + m2Mm) / 2
    if (!mirrorsClear) {
      warnings.push(
        issue(
          `M1 and M2 would overlap (${fx(gap, 0)} mm apart, need ≈${fx((spec.beam_mm + m2Mm) / 2, 0)} mm). Increase the included angle or the focal lengths.`,
          'included_deg',
          { 'Use computed angle': { included_deg: null } },
        ),
      )
    }
  }
  if (2 * spec.wl_min < spec.wl_max) {
    warnings.push(
      issue(
        `Second order of ${g(spec.wl_min)} nm lands on ${g(2 * spec.wl_min)} nm; an order-sorting (long-pass) filter is needed.`,
        'wl_max',
        { [`Limit span to ${g(spec.wl_min)}–${g(2 * spec.wl_min)} nm`]: { wl_max: 2 * spec.wl_min } },
      ),
    )
  }
  if (spec.wl_min < 0.67 * blazeNm || spec.wl_max > 1.8 * blazeNm) {
    const auto = defaultBlazeWavelength(spec.wl_min, spec.wl_max)
    const fits = 0.67 * auto <= spec.wl_min && spec.wl_max <= 1.8 * auto
    warnings.push(
      issue(
        `Span extends beyond the efficient range of a ${fx(blazeNm, 0)} nm blaze (≈${fx(0.67 * blazeNm, 0)}–${fx(1.8 * blazeNm, 0)} nm); relative efficiency ${pct(blazeEfficiency(spec.wl_min, blazeNm, m))} at ${g(spec.wl_min)} nm, ${pct(blazeEfficiency(spec.wl_max, blazeNm, m))} at ${g(spec.wl_max)} nm.`,
        'blaze_nm',
        spec.blaze_nm && fits ? { [`Blaze at ${fx(auto, 0)} nm`]: { blaze_nm: null } } : {},
      ),
    )
  }

  return { ...common, resolution, warnings, mirrors_clear: mirrorsClear, ...aberration }
}

/** Lowest stock G that meets the target and fits; else best-resolving fit; else closest to fitting. */
export function recommendGrooveDensity(spec: Spec, options = GROOVE_OPTIONS): Design {
  const designs = [...options].sort((a, b) => a - b).map((G) => analyze(spec, G, false))
  const valid = designs.filter(designOk)
  const fits = valid.filter((d) => d.spectrum_mm <= sensorMm(spec))
  const meets = fits.filter((d) => d.resolution <= spec.resolution)
  if (meets.length) return meets[0]
  const minBy = (ds: Design[], key: (d: Design) => number) => ds.reduce((a, b) => (key(b) < key(a) ? b : a))
  if (fits.length) return minBy(fits, (d) => d.resolution)
  return valid.length ? minBy(valid, (d) => d.spectrum_mm) : designs[0]
}

// Mercury–argon calibration lamp, strongest lines (nm, air).
export const HG_AR_LINES = [
  404.66, 435.83, 546.07, 576.96, 579.07, 696.54, 706.72, 738.4, 750.39, 763.51, 772.38, 794.82, 811.53, 826.45,
  842.46, 852.14, 912.3,
]

/** Position (mm) along the detector from where the centre wavelength lands; NaN if the order can't exist. */
export function sensorPosition(spec: Spec, design: Design, wl: number): number {
  const beta = diffractionAngle(wl, design.G, design.alpha, spec.m)
  const t2 = rad(design.fold2)
  return spec.f2 * (Math.tan(rad(beta - design.beta_center) - t2) + Math.tan(t2))
}

/** Linear interpolation like numpy.interp (xs ascending); `outside` beyond the ends. */
export function interp(x: number, xs: ArrayLike<number>, ys: ArrayLike<number>, outside?: number): number {
  const n = xs.length
  if (x <= xs[0]) return outside === undefined || x === xs[0] ? ys[0] : outside
  if (x >= xs[n - 1]) return outside === undefined || x === xs[n - 1] ? ys[n - 1] : outside
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] <= x) lo = mid
    else hi = mid
  }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo])
}

/**
 * Simulated detector read-out for emission lines: each line lands at
 * sensorPosition(), blurred to the design resolution (plus traced mirror blur
 * in quadrature), sampled at every pixel centre. Returns [wavelength per pixel
 * (NaN where unreachable), signal per pixel].
 */
export function detectorSignal(spec: Spec, design: Design, lines: number[], weights?: number[]) {
  const pitch = spec.pixel_um * UM_TO_MM
  const n = spec.n_pixels
  const xPix = Float64Array.from({ length: n }, (_, i) => (i - (n - 1) / 2) * pitch)
  const base = Math.hypot(design.resolution_image, design.resolution_grating) / design.rld
  const signal = new Float64Array(n)
  lines.forEach((line, k) => {
    const x0 = sensorPosition(spec, design, line)
    if (!Number.isFinite(x0)) return
    const blur = design.aberration_wl.length ? interp(line, design.aberration_wl, design.aberration_nm) : 0
    const fwhm = Math.hypot(base, blur / design.rld)
    const w = weights ? weights[k] : 1
    for (let i = 0; i < n; i++) signal[i] += w * Math.exp(-4 * Math.LN2 * ((xPix[i] - x0) / fwhm) ** 2)
  })

  // Pixel → wavelength by inverting the monotonic mapping on a fine grid.
  const span = spec.wl_max - spec.wl_min
  const lo = Math.max(spec.wl_min - span, 1)
  const hi = spec.wl_max + span
  const pairs: [number, number][] = []
  for (let i = 0; i < 4000; i++) {
    const wl = lo + ((hi - lo) * i) / 3999
    const x = sensorPosition(spec, design, wl)
    if (Number.isFinite(x)) pairs.push([x, wl])
  }
  pairs.sort((a, b) => a[0] - b[0])
  const xs = pairs.map((p) => p[0])
  const ws = pairs.map((p) => p[1])
  const wlPix = Float64Array.from(xPix, (x) => interp(x, xs, ws, NaN))
  return [wlPix, signal] as const
}

/** Lowest signal between two lines over the weaker peak: ≤ 0.81 resolved; NaN if off the sensor. */
export function lineDip(wlPix: ArrayLike<number>, signal: ArrayLike<number>, a: number, b: number): number {
  const [lo, hi] = a < b ? [a, b] : [b, a]
  let wMin = Infinity
  let wMax = -Infinity
  let between = Infinity
  const nearest = [
    [Infinity, 0],
    [Infinity, 0],
  ]
  for (let i = 0; i < wlPix.length; i++) {
    const w = wlPix[i]
    if (Number.isNaN(w)) continue
    wMin = Math.min(wMin, w)
    wMax = Math.max(wMax, w)
    if (w >= lo && w <= hi) between = Math.min(between, signal[i])
    ;[lo, hi].forEach((x, j) => {
      if (Math.abs(w - x) < nearest[j][0]) nearest[j] = [Math.abs(w - x), signal[i]]
    })
  }
  if (between === Infinity || !(wMin <= lo && hi <= wMax)) return NaN
  return between / Math.min(nearest[0][1], nearest[1][1])
}
