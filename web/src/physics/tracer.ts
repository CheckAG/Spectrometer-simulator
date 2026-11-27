/**
 * Sequential 2-D ray tracer — port of spectrometer/tracer.py.
 *
 * A ray is (x, y, angle). Rays travel as parallel arrays. Every element is a
 * line segment in its local frame (x = 0, |y| ≤ aperture/2); `theta` rotates
 * global → local (local angle = global angle + theta).
 * Grating β uses the opposite sign to optics.ts (β_tracer = −β).
 */

export interface Rays {
  x: Float64Array
  y: Float64Array
  a: Float64Array
}

export type ElementType = 'lens' | 'spherical_mirror' | 'grating' | 'sensor'

export const angleWrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a))

export function makeRays(n: number): Rays {
  return { x: new Float64Array(n), y: new Float64Array(n), a: new Float64Array(n) }
}

interface Hit {
  x: number // global
  y: number
  ahead: boolean // the ray actually travels to the element
  inside: boolean // within the aperture
  lx: number // local hit and local direction (exact mirror needs them)
  ly: number
  ux: number
  uy: number
}

export abstract class OpticalElement {
  abstract readonly type: ElementType
  readonly exact: boolean = false
  readonly c: number
  readonly s: number

  constructor(
    public aperture: number,
    public pos: [number, number],
    public theta: number,
    public name: string,
  ) {
    this.c = Math.cos(theta)
    this.s = Math.sin(theta)
  }

  toLocal(x: number, y: number): [number, number] {
    const dx = x - this.pos[0]
    const dy = y - this.pos[1]
    return [this.c * dx - this.s * dy, this.s * dx + this.c * dy]
  }

  toGlobal(lx: number, ly: number): [number, number] {
    return [this.c * lx + this.s * ly + this.pos[0], -this.s * lx + this.c * ly + this.pos[1]]
  }

  protected intersect(x: number, y: number, angle: number): Hit {
    const [px, py] = this.toLocal(x, y)
    const th = angle + this.theta
    const t = -px / Math.cos(th)
    const ly = py + t * Math.sin(th)
    const ahead = Number.isFinite(t) && t > 1e-9
    const inside = ahead && Math.abs(ly) <= this.aperture / 2 + 1e-9
    const [gx, gy] = this.toGlobal(0, ly)
    return { x: gx, y: gy, ahead, inside, lx: 0, ly, ux: Math.cos(th), uy: Math.sin(th) }
  }

  protected abstract outAngle(angle: number, lmb: number, hit: Hit): number

  /** Blocked rays get angle NaN; rays that never reach the element are NaN throughout. */
  propagate(rays: Rays, lmb: number | Float64Array): Rays {
    const n = rays.x.length
    const out = makeRays(n)
    for (let i = 0; i < n; i++) {
      const hit = this.intersect(rays.x[i], rays.y[i], rays.a[i])
      out.x[i] = hit.ahead ? hit.x : NaN
      out.y[i] = hit.ahead ? hit.y : NaN
      out.a[i] = hit.inside ? this.outAngle(rays.a[i], typeof lmb === 'number' ? lmb : lmb[i], hit) : NaN
    }
    return out
  }
}

/** Ideal thin lens / mirror: a ray at local angle a goes through (±f, f·tan a). */
export class FocalElement extends OpticalElement {
  readonly type: ElementType

  constructor(
    public f: number,
    aperture: number,
    pos: [number, number],
    theta: number,
    public reflective: boolean,
    name: string,
  ) {
    super(aperture, pos, theta, name)
    this.type = reflective ? 'spherical_mirror' : 'lens'
  }

  protected outAngle(angle: number, _lmb: number, hit: Hit): number {
    const sign = this.reflective ? -1 : 1
    const local = angle + this.theta
    const [fx, fy] = this.toGlobal(sign * this.f, this.f * Math.tan(local))
    const th = Math.atan2(fy - hit.y, fx - hit.x)
    return this.f < 0 ? th + Math.PI : th
  }
}

export const Lens = (f: number, aperture: number, pos: [number, number], theta: number, name: string) =>
  new FocalElement(f, aperture, pos, theta, false, name)
export const SphericalMirror = (f: number, aperture: number, pos: [number, number], theta: number, name: string) =>
  new FocalElement(f, aperture, pos, theta, true, name)

/** Concave sphere traced exactly (vertex at pos, concave toward local −x, R = 2f). */
export class ExactSphericalMirror extends OpticalElement {
  readonly type: ElementType = 'spherical_mirror'
  override readonly exact = true
  readonly R: number

  constructor(
    public f: number,
    aperture: number,
    pos: [number, number],
    theta: number,
    name: string,
  ) {
    super(aperture, pos, theta, name)
    this.R = 2 * f
  }

  protected override intersect(x: number, y: number, angle: number): Hit {
    const [px, py] = this.toLocal(x, y)
    const th = angle + this.theta
    const ux = Math.cos(th)
    const uy = Math.sin(th)
    const cx = px + this.R // origin relative to the centre (−R, 0)
    const b = cx * ux + py * uy
    const disc = b * b - (cx * cx + py * py - this.R * this.R)
    let t = Infinity
    for (const r of [-b - Math.sqrt(disc), -b + Math.sqrt(disc)]) {
      // vertex-side cap only, nearest hit ahead of the ray
      if (px + r * ux > -this.R && r > 1e-9 && r < t) t = r
    }
    const ahead = Number.isFinite(t)
    const lx = px + t * ux
    const ly = py + t * uy
    const inside = ahead && Math.abs(ly) <= this.aperture / 2 + 1e-9
    const [gx, gy] = ahead ? this.toGlobal(lx, ly) : [NaN, NaN]
    return { x: gx, y: gy, ahead, inside, lx, ly, ux, uy }
  }

  protected outAngle(_angle: number, _lmb: number, hit: Hit): number {
    const nx = (hit.lx + this.R) / this.R
    const ny = hit.ly / this.R
    const dot = hit.ux * nx + hit.uy * ny
    return angleWrap(Math.atan2(hit.uy - 2 * dot * ny, hit.ux - 2 * dot * nx) - this.theta)
  }
}

export class Sensor extends OpticalElement {
  readonly type: ElementType = 'sensor'
  protected outAngle(): number {
    return NaN
  }
}

/** Reflective or transmissive grating, order m. Evanescent orders → NaN. */
export class Grating extends OpticalElement {
  readonly type: ElementType = 'grating'

  constructor(
    public ngroves: number,
    aperture: number,
    pos: [number, number],
    theta: number,
    public m: number,
    public transmissive: boolean,
    name = 'Grating',
  ) {
    super(aperture, pos, theta, name)
  }

  protected outAngle(angle: number, lmb: number): number {
    const d = 1e-3 / this.ngroves
    const beta = Math.asin(Math.sin(angle + this.theta) - (this.m * lmb) / d)
    return this.transmissive ? angleWrap(beta - this.theta) : angleWrap(Math.PI - beta - this.theta)
  }
}

/** Fan of nRays from each point spanning `element`'s aperture; grouped by point. */
export function fanRays(points: [number, number][], element: OpticalElement, nRays: number): Rays {
  const hx = (element.aperture / 2) * Math.sin(element.theta)
  const hy = (element.aperture / 2) * Math.cos(element.theta)
  const out = makeRays(points.length * nRays)
  points.forEach(([px, py], p) => {
    const lo = Math.atan2(element.pos[1] + hy - py, element.pos[0] + hx - px)
    const hi = Math.atan2(element.pos[1] - hy - py, element.pos[0] - hx - px)
    for (let k = 0; k < nRays; k++) {
      const i = p * nRays + k
      out.x[i] = px
      out.y[i] = py
      out.a[i] = lo + ((hi - lo) * k) / (nRays - 1)
    }
  })
  return out
}

/** Rays at every step: [initial, after element 0, after element 1, …]. */
export function propagateRays(elements: OpticalElement[], rays: Rays, lmb: number | Float64Array): Rays[] {
  const steps = [rays]
  for (const e of elements) steps.push(e.propagate(steps[steps.length - 1], lmb))
  return steps
}

/** Rays repeated `times` (for tracing several wavelengths at once). */
export function tileRays(rays: Rays, times: number): Rays {
  const n = rays.x.length
  const out = makeRays(n * times)
  for (let t = 0; t < times; t++) {
    out.x.set(rays.x, t * n)
    out.y.set(rays.y, t * n)
    out.a.set(rays.a, t * n)
  }
  return out
}
