/** Physical layouts (4f, Czerny–Turner) built from a design, and traced — port of spectrometer/layout.py. */

import { N_RAYS, N_SOURCE_POINTS } from './config'
import type { Design, Spec } from './optics'
import {
  ExactSphericalMirror,
  Grating,
  Lens,
  type OpticalElement,
  type Rays,
  Sensor,
  SphericalMirror,
  fanRays,
  propagateRays,
  tileRays,
} from './tracer'

const rad = (d: number) => (d * Math.PI) / 180
const unit = (a: number): [number, number] => [Math.cos(a), Math.sin(a)]
const add = (p: [number, number], k: number, u: [number, number]): [number, number] => [p[0] + k * u[0], p[1] + k * u[1]]

export interface Layout {
  elements: OpticalElement[]
  slit: [number, number][] // source points across the slit
}

/**
 * Transmissive: 4f inline, slit → F1 → grating → F2 → sensor.
 * Reflective: Czerny–Turner, slit → M1 → grating → M2 → sensor.
 * `m` overrides the grating order (stray orders). `exact`: real spherical
 * mirrors, slit at M1's tangential focus, sensor on the traced best-focus line.
 */
export function buildComponents(spec: Spec, design: Design, m?: number, exact = false): Layout {
  const { f1, f2 } = spec
  const alpha = rad(design.alpha)
  const beta = rad(design.beta_center)
  const order = m ?? spec.m
  // F2 / M2 catch the beam, widened by cos β / cos α, plus its walk across the span.
  const beam2 = (spec.beam_mm * Math.cos(beta)) / Math.cos(alpha)
  const spread = f2 * Math.abs(rad(design.beta_max - design.beta_min))
  let elements: OpticalElement[]
  let slitPos: [number, number]
  let slitDir: number

  if (spec.transmissive) {
    const outDir = -(alpha + beta) // tracer β has the opposite sign
    const thetaElem = -outDir
    const pF2 = add([2 * f1, 0], f2, unit(outDir))
    const pSensor = add(pF2, f2, unit(outDir))
    const worst = rad(
      Math.max(Math.abs(design.beta_max - design.beta_center), Math.abs(design.beta_min - design.beta_center)),
    )
    const f2Aperture = 1.02 * (beam2 / Math.cos(worst) + 2 * f2 * Math.tan(worst))
    elements = [
      Lens(f1, spec.beam_mm, [f1, 0], 0, 'F1'),
      new Grating(design.G, spec.grating_mm, [2 * f1, 0], alpha, order, true),
      Lens(f2, f2Aperture, pF2, thetaElem, 'F2'),
      new Sensor(spectrumSensor(spec), pSensor, thetaElem, 'Sensor'),
    ]
    slitPos = [0, 0]
    slitDir = 0
  } else {
    // Grating at the origin, normal pointing down; M1 at α from the normal, M2 at β.
    const toM1 = alpha - Math.PI / 2
    const toM2 = beta - Math.PI / 2
    const pM1: [number, number] = [f1 * Math.cos(toM1), f1 * Math.sin(toM1)]
    const pM2: [number, number] = [f2 * Math.cos(toM2), f2 * Math.sin(toM2)]
    const Mirror = exact
      ? (f: number, ap: number, pos: [number, number], th: number, name: string) =>
          new ExactSphericalMirror(f, ap, pos, th, name)
      : SphericalMirror

    // Mirror hit by a chief ray travelling at psiIn at local incidence `off`;
    // returns it and the point on its focal plane along the chief ray.
    const fold = (f: number, pos: [number, number], psiIn: number, off: number, ap: number, name: string) => {
      const mirror = Mirror(f, ap, pos, off - psiIn, name)
      return [mirror, mirror.toGlobal(-f, f * Math.tan(off))] as const
    }
    const off1 = -rad(design.fold1) // negative: fold outward, away from the grating
    const off2 = -rad(design.fold2)
    slitDir = toM1 + 2 * off1
    const [m1] = fold(f1, pM1, slitDir, off1, spec.beam_mm, 'M1')
    // Ideal: slit on M1's front focal plane; exact sphere: tangential focus f·cos(off).
    slitPos = add(pM1, -(exact ? f1 * Math.cos(off1) : f1 / Math.cos(off1)), unit(slitDir))
    const [m2, focal] = fold(f2, pM2, toM2, off2, (beam2 + spread) / Math.cos(off2), 'M2')
    let sensorPos: [number, number] = focal
    let sensorTheta = m2.theta
    if (exact) {
      // Sensor through the best foci of the span ends, centred on the centre focus (design order).
      const g1 = new Grating(design.G, spec.grating_mm, [0, 0], -Math.PI / 2, spec.m, false)
      const foci = [spec.wl_min, (spec.wl_min + spec.wl_max) / 2, spec.wl_max].map((wl) =>
        bestFocus([m1, g1, m2], slitPos, wl),
      )
      sensorPos = foci[1]
      sensorTheta = Math.PI / 2 - Math.atan2(foci[2][1] - foci[0][1], foci[2][0] - foci[0][0])
    }
    elements = [
      m1,
      new Grating(design.G, spec.grating_mm, [0, 0], -Math.PI / 2, order, false),
      m2,
      new Sensor(spectrumSensor(spec), sensorPos, sensorTheta, 'Sensor'),
    ]
  }

  // Slit: short line perpendicular to the chief ray leaving it.
  const half = (design.slit_um * 1e-3) / 2
  const across = unit(slitDir + Math.PI / 2)
  const slit = Array.from({ length: N_SOURCE_POINTS }, (_, i) =>
    add(slitPos, half - (2 * half * i) / (N_SOURCE_POINTS - 1), across),
  )
  return { elements, slit }
}

const spectrumSensor = (spec: Spec) => spec.n_pixels * spec.pixel_um * 1e-3

/** Least-squares point nearest every ray of a fan from `point` leaving the last element. */
export function bestFocus(elements: OpticalElement[], point: [number, number], wl: number, nRays = 101): [number, number] {
  const steps = propagateRays(elements, fanRays([point], elements[0], nRays), wl * 1e-9)
  const out = steps[steps.length - 1]
  // Σ (I − u uᵀ) x = Σ (I − u uᵀ) p
  let a = 0
  let b = 0
  let d = 0
  let rx = 0
  let ry = 0
  let count = 0
  for (let i = 0; i < nRays; i++) {
    const [x, y, an] = [out.x[i], out.y[i], out.a[i]]
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(an)) continue
    const ux = Math.cos(an)
    const uy = Math.sin(an)
    const pxx = 1 - ux * ux
    const pxy = -ux * uy
    const pyy = 1 - uy * uy
    a += pxx
    b += pxy
    d += pyy
    rx += pxx * x + pxy * y
    ry += pxy * x + pyy * y
    count++
  }
  if (count < 2) return [NaN, NaN]
  const det = a * d - b * b
  return [(d * rx - b * ry) / det, (a * ry - b * rx) / det]
}

/** Local position along the sensor (mm), unclipped. */
const sensorY = (sensor: OpticalElement, x: number, y: number) => sensor.toLocal(x, y)[1]

/**
 * Spectral blur (nm, 2.355·σ) the mirrors add, per wavelength: a slit point
 * traced through exact mirrors, RMS spot width over the traced dispersion.
 * Transmissive layouts use ideal lenses: zeros.
 */
export function aberrationBlur(spec: Spec, design: Design, nWl = 9, nRays = 201): [number[], number[]] {
  const wls = Array.from({ length: nWl }, (_, i) => spec.wl_min + ((spec.wl_max - spec.wl_min) * i) / (nWl - 1))
  if (spec.transmissive) return [wls, wls.map(() => 0)]
  const { elements, slit } = buildComponents(spec, design, undefined, true)
  const mid: [number, number] = [
    slit.reduce((s, p) => s + p[0], 0) / slit.length,
    slit.reduce((s, p) => s + p[1], 0) / slit.length,
  ]
  const rays = fanRays([mid], elements[0], nRays)
  const lmb = Float64Array.from({ length: nWl * nRays }, (_, i) => wls[Math.floor(i / nRays)] * 1e-9)
  const steps = propagateRays(elements, tileRays(rays, nWl), lmb)
  const end = steps[steps.length - 1]
  const sensor = elements[elements.length - 1]
  const mean: number[] = []
  const std: number[] = []
  for (let w = 0; w < nWl; w++) {
    const ys: number[] = []
    for (let k = 0; k < nRays; k++) {
      const i = w * nRays + k
      const y = sensorY(sensor, end.x[i], end.y[i])
      if (Number.isFinite(y)) ys.push(y)
    }
    const mu = ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : NaN
    mean.push(mu)
    std.push(ys.length ? Math.sqrt(ys.reduce((s, v) => s + (v - mu) ** 2, 0) / ys.length) : NaN)
  }
  if (mean.every(Number.isNaN)) return [wls, wls.map(() => NaN)]
  // numpy.gradient: one-sided at the ends, central inside (uniform spacing).
  const dxdl = mean.map((_, i) => {
    if (i === 0) return (mean[1] - mean[0]) / (wls[1] - wls[0])
    if (i === nWl - 1) return (mean[i] - mean[i - 1]) / (wls[i] - wls[i - 1])
    return (mean[i + 1] - mean[i - 1]) / (wls[i + 1] - wls[i - 1])
  })
  return [wls, std.map((s, i) => (2.355 * s) / Math.abs(dxdl[i]))]
}

export interface Trace {
  elements: OpticalElement[]
  steps: Rays[] // initial rays, then after each element; rays grouped by wavelength
  raysPerWl: number
}

/** Trace every wavelength in one call. */
export function trace(spec: Spec, design: Design, wavelengths: number[], m?: number, exact = false): Trace {
  const { elements, slit } = buildComponents(spec, design, m, exact)
  const rays = fanRays(slit, elements[0], N_RAYS)
  const n = rays.x.length
  const lmb = Float64Array.from({ length: n * wavelengths.length }, (_, i) => wavelengths[Math.floor(i / n)] * 1e-9)
  return { elements, steps: propagateRays(elements, tileRays(rays, wavelengths.length), lmb), raysPerWl: n }
}

/** Local positions (mm) on the sensor; NaN for rays that miss it. */
export function sensorHits(t: Trace): Float64Array {
  const sensor = t.elements[t.elements.length - 1]
  const end = t.steps[t.steps.length - 1]
  return Float64Array.from(end.x, (x, i) => {
    const y = sensorY(sensor, x, end.y[i])
    return Math.abs(y) <= sensor.aperture / 2 ? y : NaN
  })
}

