/**
 * The optical bench: the traced instrument drawn on a breadboard, and the
 * main way to edit it. Drag F1/M1 or F2/M2 along their beam to change the
 * focal lengths, turn the grating dial to change the included angle.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { ELEMENT_COLOR, SLIT_COLOR, wavelengthRgbTuple } from './color'
import { N_WAVELENGTH_SAMPLES } from './physics/config'
import { buildComponents, trace } from './physics/layout'
import type { Design, Spec } from './physics/optics'
import type { OpticalElement, Rays } from './physics/tracer'

type P = [number, number]
type DragKind = 'f1' | 'f2' | 'angle'

const rad = (d: number) => (d * Math.PI) / 180
const unit = (a: number): P => [Math.cos(a), Math.sin(a)]
const dot = (a: P, b: P) => a[0] * b[0] + a[1] * b[1]
const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
const nanMean = (v: number[]) => {
  const ok = v.filter(Number.isFinite)
  return ok.length ? ok.reduce((s, x) => s + x, 0) / ok.length : NaN
}

/** Polylines for rays [i0, i1) from step `start`, extended by `reach` while still travelling. */
function rayLines(steps: Rays[], i0: number, i1: number, reach: number, start = 0): P[][] {
  const out: P[][] = []
  const last = steps.length - 1
  for (let i = i0; i < i1; i++) {
    let n = steps.findIndex((s) => Number.isNaN(s.x[i]))
    if (n < 0) n = steps.length
    const pts: P[] = []
    for (let k = start; k < n; k++) pts.push([steps[k].x[i], steps[k].y[i]])
    if (n === steps.length && !Number.isNaN(steps[last].a[i])) {
      pts.push([steps[last].x[i] + reach * Math.cos(steps[last].a[i]), steps[last].y[i] + reach * Math.sin(steps[last].a[i])])
    }
    if (pts.length > 1) out.push(pts)
  }
  return out
}

/** Element outline in its local frame; `back` is the exit / non-reflecting side. */
function glyph(c: OpticalElement, back: number, transmissive: boolean): [number[], number[]] {
  const h = c.aperture / 2
  const t = linspace(-h, h, 41)
  const rev = [...t].reverse()
  if (c.type === 'lens') {
    const w = t.map((v) => 0.06 * c.aperture * (1 - (v / h) ** 2) + 0.4)
    return [[...w, ...w.map((v) => -v).reverse()], [...t, ...rev]]
  }
  if (c.type === 'spherical_mirror') {
    const f = Math.abs((c as unknown as { f: number }).f)
    const face = t.map((v) => (-back * (v ** 2 - (c.exact ? 0 : h ** 2))) / (4 * f))
    return [[...face, ...t.map(() => back * 2)], [...t, ...rev]]
  }
  if (c.type === 'grating') {
    const n = 16
    const ty = linspace(-h, h, 2 * n + 1)
    const face = transmissive ? back : -back // grooves on the working face
    return [[...ty.map((_, i) => (i % 2 ? face * 1.2 : 0)), -face * 3, -face * 3], [...ty, h, -h]]
  }
  return [[0, 0, back * 1.5, back * 1.5], [-h, h, h, -h]] // sensor package
}

const polygon = (c: OpticalElement, gx: number[], gy: number[]) =>
  gx.map((x, i) => {
    const [px, py] = c.toGlobal(x, gy[i])
    return `${px.toFixed(2)},${(-py).toFixed(2)}`
  }).join(' ')

export interface BenchProps {
  spec: Spec
  design: Design
  exact: boolean
  stray: boolean
  highlight: Set<number> // element indices to flag (0 collimator, 1 grating, 2 focuser, 3 sensor)
  onF1: (v: number) => void
  onF2: (v: number) => void
  onAngle: (v: number) => void
}

export default function Bench({ spec, design, exact, stray, highlight, onF1, onF2, onAngle }: BenchProps) {
  const svg = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<{ kind: DragKind; box: number[]; x0: number; angle0: number } | null>(null)

  const scene = useMemo(() => {
    const { elements, slit } = buildComponents(spec, design, undefined, exact)
    const wls = linspace(spec.wl_min, spec.wl_max, N_WAVELENGTH_SAMPLES)
    const slitMid: P = [(slit[0][0] + slit[1][0]) / 2, (slit[0][1] + slit[1][1]) / 2]
    const pts: P[] = [...elements.map((e) => e.pos), slitMid]
    const xs = pts.map((p) => p[0])
    const ys = pts.map((p) => p[1])
    const pad = Math.max(spec.grating_mm * 0.55, 18)
    const w = Math.max(...xs) - Math.min(...xs) + 2 * pad
    const h = Math.max(...ys) - Math.min(...ys) + 2 * pad
    const head = 0.12 * Math.max(w, h) // headroom for the readouts overlaid at the top
    const box = [Math.min(...xs) - pad, -(Math.max(...ys) + pad) - head, w, h + head]
    const reach = 1.5 * Math.max(box[2], box[3])
    const white = trace(spec, design, wls.slice(0, 1), undefined, exact)
    const orders = [spec.m, ...(stray ? [0, 2] : [])].map((m) => {
      const t = trace(spec, design, wls, m, exact)
      return { m, paths: wls.map((wl, k) => ({ wl, lines: rayLines(t.steps, k * t.raysPerWl, (k + 1) * t.raysPerWl, reach, 2) })) }
    })
    const whiteLines = rayLines(white.steps.slice(0, 3), 0, white.steps[0].x.length, 0)
    const parts = elements.map((c, k) => {
      const arriving = Array.from(white.steps[k].a)
      const a = Math.atan2(nanMean(arriving.map(Math.sin)), nanMean(arriving.map(Math.cos)))
      const [n1x, n1y] = c.toGlobal(1, 0)
      const back = Number.isNaN(a) || (n1x - c.pos[0]) * Math.cos(a) + (n1y - c.pos[1]) * Math.sin(a) >= 0 ? 1 : -1
      const [gx, gy] = glyph(c, back, spec.transmissive)
      // Label on the far side of the element from the beam centre.
      const h = c.aperture / 2
      const ends = [c.toGlobal(0, -h - 8), c.toGlobal(0, h + 8)]
      const centre: P = [box[0] + box[2] / 2, -(box[1] + box[3] / 2)]
      const d = (p: P) => Math.hypot(p[0] - centre[0], p[1] - centre[1])
      const farEnd = d(ends[0]) > d(ends[1]) ? 0 : 1
      // Czerny–Turner grating: both ends are busy (sensor, slit), so label it from behind.
      const label: P = !spec.transmissive && c.type === 'grating' ? [c.pos[0] - 16, c.pos[1] + 13] : ends[farEnd]
      return { c, points: polygon(c, gx, gy), label, other: farEnd ? -1 : 1 }
    })
    return { box, orders, whiteLines, parts, slit, slitMid }
  }, [spec, design, exact, stray])

  const box = drag?.box ?? scene.box // freeze the view while dragging so the pointer maps stably
  const canvas = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState<[number, number]>([0, 0])
  useEffect(() => {
    const el = svg.current!
    const ro = new ResizeObserver(() => setSize([el.clientWidth, el.clientHeight]))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Light is drawn on a canvas with additive blending ('lighter'): overlapping
  // wavelengths add up like real light, cheaply. Same mapping as the SVG viewBox (meet, centred).
  useEffect(() => {
    const cv = canvas.current
    const [W, H] = size
    if (!cv || !W || !H) return
    const dpr = window.devicePixelRatio || 1
    cv.width = W * dpr
    cv.height = H * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)
    const k = Math.min(W / box[2], H / box[3])
    const ox = (W - box[2] * k) / 2
    const oy = (H - box[3] * k) / 2
    const map = ([x, y]: P): P => [ox + (x - box[0]) * k, oy + (-y - box[1]) * k]
    const stroke = (lines: P[][], style: string, width: number, dash: number[] = []) => {
      ctx.strokeStyle = style
      ctx.lineWidth = width
      ctx.setLineDash(dash)
      ctx.beginPath()
      for (const line of lines) line.forEach((p, i) => (i ? ctx.lineTo(...map(p)) : ctx.moveTo(...map(p))))
      ctx.stroke()
    }
    ctx.globalCompositeOperation = 'lighter'
    ctx.lineCap = 'round'
    // Brightness per ray falls with the rays sharing a wavelength, so dense fans stay coloured.
    const gain = Math.min(1, 8 / Math.max(1, scene.whiteLines.length / 2))
    stroke(scene.whiteLines, `rgba(235,238,245,${0.1 * gain})`, 3.5)
    stroke(scene.whiteLines, `rgba(235,238,245,${0.35 * gain})`, 0.8)
    for (const { m, paths } of scene.orders) {
      const main = m === spec.m
      const dash = main ? [] : m === 0 ? [2, 4] : [8, 5]
      for (const { wl, lines } of paths) {
        const [r, g, b] = wavelengthRgbTuple(wl).map((v) => Math.round(v * 255))
        const a = (main ? 1 : 0.45) * gain
        stroke(lines, `rgba(${r},${g},${b},${0.07 * a})`, 5, dash) // glow
        stroke(lines, `rgba(${r},${g},${b},${0.5 * a})`, 1, dash) // core
      }
    }
  }, [scene, box, size, spec.m])
  const u = Math.max(box[2], box[3]) / 100 // one "unit" for strokes and text
  const alpha = rad(design.alpha)
  const beta = rad(design.beta_center)

  const toPhysics = (e: React.PointerEvent): P => {
    const ctm = svg.current!.getScreenCTM()!.inverse()
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm)
    return [p.x, -p.y]
  }
  const onMove = (e: React.PointerEvent) => {
    if (!drag) return
    const p = toPhysics(e)
    if (drag.kind === 'angle') {
      onAngle(Math.round(Math.min(90, Math.max(0, drag.angle0 + (p[0] - drag.x0) / (box[2] / 140)))))
    } else if (drag.kind === 'f1') {
      const v = spec.transmissive ? p[0] : dot(p, unit(alpha - Math.PI / 2))
      onF1(Math.round(Math.min(200, Math.max(10, v))))
    } else {
      const v = spec.transmissive ? dot([p[0] - 2 * spec.f1, p[1]], unit(-(alpha + beta))) : dot(p, unit(beta - Math.PI / 2))
      onF2(Math.round(Math.min(200, Math.max(10, v))))
    }
  }
  const start = (kind: DragKind) => (e: React.PointerEvent) => {
    ;(e.target as Element).setPointerCapture(e.pointerId)
    setDrag({ kind, box: scene.box, x0: toPhysics(e)[0], angle0: design.included_deg })
  }

  const grating = scene.parts[1].c
  // Dial: 4f — past the grating end opposite its label; Czerny–Turner — behind the grating (light never goes there).
  const dialAt: P = spec.transmissive
    ? grating.toGlobal(0, scene.parts[1].other * (grating.aperture / 2 + 10))
    : [grating.pos[0] + 16, grating.pos[1] + 16]
  const fmtF = (v: number) => `${v.toFixed(0)} mm`

  return (
    <div className="bench-stage" style={{ height: size[0] ? Math.min(640, Math.max(280, (size[0] * box[3]) / box[2])) : undefined }}>
    <svg className="bench-holes" viewBox={box.map((v) => v.toFixed(2)).join(' ')} aria-hidden>
      <defs>
        {/* Breadboard: M6 holes on a 25 mm pitch, aligned to the optical axis origin. */}
        <pattern id="holes" width="25" height="25" patternUnits="userSpaceOnUse" x="-12.5" y="-12.5">
          <circle cx="12.5" cy="12.5" r="1.1" className="hole" />
        </pattern>
      </defs>
      <rect x={box[0]} y={box[1]} width={box[2]} height={box[3]} fill="url(#holes)" />
    </svg>
    <canvas ref={canvas} className="bench-light" aria-hidden />
    <svg
      ref={svg}
      className={`bench ${drag ? 'dragging' : ''}`}
      viewBox={box.map((v) => v.toFixed(2)).join(' ')}
      onPointerMove={onMove}
      onPointerUp={() => setDrag(null)}
      onPointerCancel={() => setDrag(null)}
      role="img"
      aria-label={`Ray trace of the ${spec.transmissive ? '4f lens' : 'Czerny–Turner mirror'} spectrometer`}
    >

      {scene.parts.map(({ c, points, label }, k) => {
        const color = ELEMENT_COLOR[c.type]
        const f = (c as unknown as { f?: number }).f
        const detail = c.type === 'grating' ? `${design.G} l/mm` : c.type === 'sensor' ? `${spec.n_pixels} px` : fmtF(Math.abs(f ?? 0))
        return (
          <g key={c.name} className={`part ${highlight.has(k) ? 'flagged' : ''}`}>
            <polygon points={points} fill={color} fillOpacity={0.28} stroke={color} strokeWidth={u * 0.18} />
            <text x={label[0]} y={-label[1]} fontSize={u * 2.1} fill={color} textAnchor="middle" className="part-label">
              <tspan fontWeight={700}>{c.name}</tspan>
              <tspan x={label[0]} dy={u * 2.3} fontSize={u * 1.6} className="part-detail">{detail}</tspan>
            </text>
            {(k === 0 || k === 2) && (
              <circle
                cx={c.pos[0]}
                cy={-c.pos[1]}
                r={u * 1.6}
                className="handle"
                stroke={color}
                strokeWidth={u * 0.25}
                onPointerDown={start(k === 0 ? 'f1' : 'f2')}
              >
                <title>{`Drag along the beam to change ${k === 0 ? 'f1' : 'f2'}`}</title>
              </circle>
            )}
          </g>
        )
      })}

      {/* Grating dial: drag sideways to turn the included angle. */}
      <g className="dial" onPointerDown={start('angle')}>
        <circle cx={dialAt[0]} cy={-dialAt[1]} r={u * 2.2} stroke={ELEMENT_COLOR.grating} strokeWidth={u * 0.25} />
        <line
          x1={dialAt[0]}
          y1={-dialAt[1]}
          x2={dialAt[0] + u * 2.2 * Math.sin(rad(design.included_deg * 3))}
          y2={-dialAt[1] - u * 2.2 * Math.cos(rad(design.included_deg * 3))}
          stroke={ELEMENT_COLOR.grating}
          strokeWidth={u * 0.3}
        />
        <text x={dialAt[0] + u * 3.2} y={-dialAt[1] + u * 0.7} fontSize={u * 1.8} fill={ELEMENT_COLOR.grating}>
          2K {design.included_deg.toFixed(0)}°
        </text>
        <title>Drag sideways to change the included angle</title>
      </g>

      {/* Slit: two jaws across the beam (the µm gap is far too small to draw). */}
      {(() => {
        const [sx, sy] = scene.slitMid
        let ax = scene.slit[0][0] - scene.slit[1][0]
        let ay = scene.slit[0][1] - scene.slit[1][1]
        const len = Math.hypot(ax, ay)
        ;[ax, ay] = len ? [ax / len, ay / len] : [0, 1]
        const g = 0.4
        const j = u * 3
        return (
          <g className="slit">
            <line x1={sx + g * ax} y1={-(sy + g * ay)} x2={sx + (g + j) * ax} y2={-(sy + (g + j) * ay)} stroke={SLIT_COLOR} strokeWidth={u * 0.6} />
            <line x1={sx - g * ax} y1={-(sy - g * ay)} x2={sx - (g + j) * ax} y2={-(sy - (g + j) * ay)} stroke={SLIT_COLOR} strokeWidth={u * 0.6} />
            <text x={sx - (j + u * 3) * ax} y={-(sy - (j + u * 3) * ay)} fontSize={u * 2.1} fill={SLIT_COLOR} textAnchor="middle" fontWeight={700}>
              Slit
              <tspan x={sx - (j + u * 3) * ax} dy={u * 2.3} fontSize={u * 1.6} fontWeight={400}>{`${design.slit_um.toFixed(1)} µm`}</tspan>
            </text>
            <title>{`Slit · ${design.slit_um.toFixed(1)} µm`}</title>
          </g>
        )
      })()}

      {/* 25 mm scale bar */}
      <g className="scalebar" transform={`translate(${box[0] + u * 3} ${box[1] + box[3] - u * 3})`}>
        <line x1={0} y1={0} x2={25} y2={0} strokeWidth={u * 0.25} />
        <text x={12.5} y={-u * 1} fontSize={u * 1.5} textAnchor="middle">25 mm</text>
      </g>
    </svg>
    </div>
  )
}
