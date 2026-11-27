/**
 * "How a spectrometer works" — five small animated, interactive films.
 * Scenes are schematic but computed from the real equations in physics/optics.
 */

import { type ReactNode, useEffect, useRef, useState } from 'react'
import { wavelengthRgb } from './color'
import { blazeEfficiency, diffractionAngle } from './physics/optics'

const rad = (d: number) => (d * Math.PI) / 180
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Seconds since start while `playing`; frozen otherwise. */
function useClock(playing: boolean) {
  const [t, setT] = useState(0)
  const base = useRef<number | null>(null)
  const acc = useRef(0)
  useEffect(() => {
    if (!playing) {
      base.current = null
      return
    }
    let raf = 0
    const tick = (now: number) => {
      if (base.current === null) base.current = now - acc.current * 1000
      acc.current = (now - base.current) / 1000
      setT(acc.current)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])
  return t
}

/** Plays only while on screen (and not under reduced motion unless asked). */
function useOnScreen(ref: React.RefObject<Element | null>) {
  const [on, setOn] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => setOn(e.isIntersecting), { threshold: 0.25 })
    io.observe(el)
    return () => io.disconnect()
  }, [ref])
  return on
}

function Chapter(props: { n: number; title: string; lede: ReactNode; children: (t: number) => ReactNode; controls?: ReactNode }) {
  const ref = useRef<HTMLElement>(null)
  const visible = useOnScreen(ref)
  const [paused, setPaused] = useState(reducedMotion())
  const t = useClock(visible && !paused)
  return (
    <article className="chapter" ref={ref}>
      <div className="chapter-text">
        <span className="chapter-n">{String(props.n).padStart(2, '0')}</span>
        <h3>{props.title}</h3>
        <div className="chapter-lede">{props.lede}</div>
        {props.controls && <div className="chapter-controls">{props.controls}</div>}
      </div>
      <div className="stage">
        <svg viewBox="0 0 400 240" role="img" aria-label={props.title}>
          {props.children(t)}
        </svg>
        <button className="play" onClick={() => setPaused((p) => !p)} aria-label={paused ? 'Play' : 'Pause'}>
          {paused ? '▶' : '❚❚'}
        </button>
      </div>
    </article>
  )
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (v: number) => void; color?: string }) {
  return (
    <label className="ctl">
      <span>
        {props.label} <b style={{ color: props.color }}>{props.value.toFixed(props.step < 1 ? 1 : 0)} {props.unit}</b>
      </span>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} onChange={(e) => props.onChange(Number(e.target.value))} />
    </label>
  )
}

/* ── 1. The journey of light ─────────────────────────────────────────────── */

const WLS = [420, 460, 500, 540, 580, 620, 660, 700]
const JOURNEY_STEPS: [number, string, string][] = [
  [0, 'Slit', 'A narrow slit lets in a thin line of the light — a well-defined source.'],
  [2, 'F1', 'The first lens makes the rays parallel, so every ray meets the grating at the same angle.'],
  [4, 'Grating', 'The grating sends each colour off at its own angle — white light fans into a spectrum.'],
  [6, 'F2', 'The second lens focuses each angle — each colour — to its own spot.'],
  [8, 'Sensor', 'Every pixel now measures one narrow band of colour: that is the spectrum.'],
]

function Journey(t: number) {
  const T = 10
  const phase = t % T
  const step = [...JOURNEY_STEPS].reverse().find(([s]) => phase >= s)!
  // Geometry (schematic): slit → F1 (collimate) → grating (disperse) → F2 (focus) → sensor.
  const slit = { x: 40, y: 120 }
  const xF1 = 115
  const xG = 195
  const xF2 = 285
  const xS = 360
  const heights = [-26, -13, 0, 13, 26]
  const disp = (wl: number) => (wl - 560) * 0.0016 // fan angle per colour (rad)
  const focusY = (wl: number) => 120 + (xS - xF2) * Math.tan(disp(wl)) * 2.2
  const path = (h: number, wl: number | null) => {
    const pts: [number, number][] = [[slit.x, slit.y], [xF1, 120 + h], [xG, 120 + h]]
    if (wl !== null) {
      const a = disp(wl)
      pts.push([xF2, 120 + h + (xF2 - xG) * Math.tan(a)])
      pts.push([xS, focusY(wl)])
    }
    return pts
  }
  // Photon position along a polyline at fraction s ∈ [0, 1].
  const along = (pts: [number, number][], s: number): [number, number] => {
    const seg = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]))
    let d = s * seg.reduce((a, b) => a + b, 0)
    for (let i = 0; i < seg.length; i++) {
      if (d <= seg[i]) return [pts[i][0] + ((pts[i + 1][0] - pts[i][0]) * d) / seg[i], pts[i][1] + ((pts[i + 1][1] - pts[i][1]) * d) / seg[i]]
      d -= seg[i]
    }
    return pts[pts.length - 1]
  }
  const reached = Math.min(1, phase / 8) // how far the light has got this cycle
  const lit = (name: string) => (step[1] === name ? 'lit' : '')
  const photons: ReactNode[] = []
  WLS.forEach((wl, w) =>
    heights.forEach((h, k) => {
      const pts = path(h, wl)
      for (let j = 0; j < 3; j++) {
        const s = ((t * 0.18 + j / 3 + (w * 5 + k) * 0.037) % 1) * 1
        if (s > reached) continue
        const [x, y] = along(pts, s)
        const white = x < xG
        photons.push(<circle key={`${wl}-${h}-${j}`} cx={x} cy={y} r={white ? 1.3 : 1.6} fill={white ? '#f5f7ff' : wavelengthRgb(wl)} opacity={white ? 0.5 : 0.9} />)
      }
    }),
  )
  return (
    <>
      <defs>
        <radialGradient id="bulb">
          <stop offset="0" stopColor="#fff8e0" />
          <stop offset="1" stopColor="#fff8e0" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx={14} cy={120} r={16} fill="url(#bulb)" opacity={0.6 + 0.2 * Math.sin(t * 3)} />
      {/* faint ray paths */}
      {WLS.map((wl) => heights.map((h) => <polyline key={`${wl}${h}`} points={path(h, wl).map((p) => p.join(',')).join(' ')} fill="none" stroke={wavelengthRgb(wl)} strokeOpacity={0.07} />))}
      <g className={`el ${lit('Slit')}`}>
        <line x1={slit.x} y1={98} x2={slit.x} y2={117} stroke="#f2b84b" strokeWidth={3} />
        <line x1={slit.x} y1={123} x2={slit.x} y2={142} stroke="#f2b84b" strokeWidth={3} />
      </g>
      <g className={`el ${lit('F1')}`}>
        <ellipse cx={xF1} cy={120} rx={4} ry={34} fill="#7fb2ff" fillOpacity={0.25} stroke="#7fb2ff" />
      </g>
      <g className={`el ${lit('Grating')}`}>
        <rect x={xG - 2} y={84} width={4} height={72} fill="#d38bff" fillOpacity={0.3} stroke="#d38bff" />
        {Array.from({ length: 12 }, (_, i) => <line key={i} x1={xG + 2} x2={xG + 5} y1={87 + i * 6} y2={90 + i * 6} stroke="#d38bff" />)}
      </g>
      <g className={`el ${lit('F2')}`}>
        <ellipse cx={xF2} cy={120} rx={4} ry={44} fill="#7fb2ff" fillOpacity={0.25} stroke="#7fb2ff" />
      </g>
      {photons}
      <g className={`el ${lit('Sensor')}`}>
        {Array.from({ length: 24 }, (_, i) => {
          const y = 90 + i * 2.6
          const wl = WLS.reduce((b, w) => (Math.abs(focusY(w) - y) < Math.abs(focusY(b) - y) ? w : b))
          const on = Math.abs(focusY(wl) - y) < 2.2 && phase > 7.6
          return <rect key={i} x={xS} y={y} width={7} height={2.2} fill={on ? wavelengthRgb(wl) : '#1b2030'} />
        })}
      </g>
      {[['Slit', slit.x], ['F1', xF1], ['Grating', xG], ['F2', xF2], ['Sensor', xS + 4]].map(([n, x]) => (
        <text key={n} x={x as number} y={200} textAnchor="middle" className={`tag ${step[1] === n ? 'on' : ''}`}>
          {n}
        </text>
      ))}
      <text x={200} y={228} textAnchor="middle" className="caption">{step[2]}</text>
    </>
  )
}

/* ── 2. Inside the grating: wavelets add up ──────────────────────────────── */

function GratingScene(t: number, wl: number, G: number) {
  const alpha = 18
  const beta = diffractionAngle(wl, G, alpha, 1) // optics convention
  const y0 = 150
  const pitch = 26 // schematic groove spacing (px)
  const grooves = Array.from({ length: 13 }, (_, i) => 44 + i * pitch)
  const lam = pitch * (wl * 1e-6 * G) // wavelength drawn to the same scale as the groove pitch
  const speed = 18
  const color = wavelengthRgb(wl)
  // Incoming wavefronts travel down-right at α from the normal (from the left).
  const kin: [number, number] = [Math.sin(rad(alpha)), Math.cos(rad(alpha))]
  const fronts = Array.from({ length: 7 }, (_, i) => ((t * speed) % lam) + i * lam - 7 * lam)
  // Outgoing direction: β on the same side as α → goes up-left (reflection grating).
  const out: [number, number] = [-Math.sin(rad(beta)), -Math.cos(rad(beta))]
  const ok = Number.isFinite(beta)
  return (
    <>
      <defs>
        <clipPath id="above">
          <rect x={0} y={0} width={400} height={y0} />
        </clipPath>
      </defs>
      <g clipPath="url(#above)">
        {/* incoming plane wavefronts */}
        {fronts.map((d, i) => {
          const cx = 200 + kin[0] * d - kin[0] * 0
          const cy = y0 + kin[1] * d
          return <line key={`in${i}`} x1={cx - kin[1] * 220} y1={cy + kin[0] * 220} x2={cx + kin[1] * 220} y2={cy - kin[0] * 220} stroke="#e8ecf5" strokeOpacity={0.35} />
        })}
        {/* Huygens wavelets from each groove, delayed by the incoming wave's arrival */}
        {grooves.map((x, i) =>
          [0, 1, 2, 3, 4, 5].map((k) => {
            const delay = ((x - 200) * kin[0]) / speed
            const r = ((t - delay) * speed) % lam + k * lam
            return r > 0 && r < 6 * lam ? <circle key={`${i}-${k}`} cx={x} cy={y0} r={r} fill="none" stroke={color} strokeOpacity={0.22 * (1 - r / (6 * lam))} /> : null
          }),
        )}
        {/* the wavefront they build: plane waves along β */}
        {ok &&
          [1, 2, 3, 4, 5].map((k) => {
            const d = ((t * speed) % lam) + k * lam
            const cx = 200 + out[0] * d
            const cy = y0 + out[1] * d
            return <line key={`out${k}`} x1={cx - out[1] * 200} y1={cy + out[0] * 200} x2={cx + out[1] * 200} y2={cy - out[0] * 200} stroke={color} strokeWidth={2} strokeOpacity={0.85} />
          })}
      </g>
      {/* grating surface with grooves */}
      <rect x={20} y={y0} width={360} height={14} fill="#d38bff" fillOpacity={0.18} />
      {grooves.map((x) => <path key={x} d={`M${x - 10} ${y0} L${x} ${y0 - 5} L${x + 2} ${y0}`} fill="#d38bff" fillOpacity={0.5} />)}
      <line x1={200} y1={y0} x2={200} y2={40} stroke="#59627a" strokeDasharray="3 4" />
      <text x={204} y={50} className="tag">normal</text>
      <path d={`M200 ${y0} L${200 - 70 * kin[0]} ${y0 - 70 * kin[1]}`} stroke="#e8ecf5" strokeOpacity={0.6} markerEnd="" />
      <text x={200 - 82 * kin[0]} y={y0 - 82 * kin[1]} className="tag" textAnchor="middle">α {alpha}°</text>
      {ok && <path d={`M200 ${y0} L${200 + 90 * out[0]} ${y0 + 90 * out[1]}`} stroke={color} strokeWidth={2} />}
      {ok && <text x={200 + 104 * out[0]} y={y0 + 104 * out[1]} className="tag" textAnchor="middle" fill={color}>β {beta.toFixed(1)}°</text>}
      <text x={200} y={192} textAnchor="middle" className="eq">
        1 × {wl} nm × {G}/mm = sin {alpha}° + sin β
      </text>
      <text x={200} y={214} textAnchor="middle" className="caption">
        {ok ? `→ β = ${beta.toFixed(1)}°. Longer wavelengths bend further.` : 'No solution: this colour cannot leave the grating in order 1.'}
      </text>
    </>
  )
}

/* ── 3. Resolution: when two lines merge ────────────────────────────────── */

function ResolutionScene(t: number, sep: number, width: number) {
  const n = 48
  const c = 589
  const pix = (i: number) => c - 4 + (8 * i) / (n - 1) // nm per pixel centre
  const line = (x: number, at: number) => Math.exp(-4 * Math.LN2 * ((x - at) / width) ** 2)
  const sweep = 20 + ((t * 60) % 360) // read-out scanning across the pixels
  const sig = Array.from({ length: n }, (_, i) => line(pix(i), c - sep / 2) + line(pix(i), c + sep / 2))
  const peak = Math.max(...sig)
  const mid = sig.slice(Math.floor(n / 2) - 1, Math.ceil(n / 2) + 1).reduce((a, b) => Math.min(a, b))
  const top = Math.max(line(c - sep / 2, c - sep / 2) + line(c - sep / 2, c + sep / 2), 1e-9)
  const dip = mid / top
  const resolved = dip <= 0.81
  const W = 360 / n
  return (
    <>
      {sig.map((s, i) => (
        <rect key={i} x={20 + i * W} y={150 - (s / peak) * 110} width={W - 1.5} height={(s / peak) * 110} fill={wavelengthRgb(pix(i))} fillOpacity={0.25 + 0.75 * (s / peak)} rx={1} />
      ))}
      {[c - sep / 2, c + sep / 2].map((x) => {
        const px = 20 + ((x - (c - 4)) / 8) * 360
        return <line key={x} x1={px} x2={px} y1={30} y2={150} stroke="#e8ecf5" strokeDasharray="2 4" strokeOpacity={0.6} />
      })}
      <line x1={20} x2={380} y1={150 - (0.81 * top / peak) * 110} y2={150 - (0.81 * top / peak) * 110} stroke="#59627a" strokeDasharray="4 4" />
      <text x={382} y={150 - (0.81 * top / peak) * 110 - 4} textAnchor="end" className="tag">Rayleigh 81 %</text>
      <rect x={20} y={156} width={360} height={10} fill="#1b2030" />
      <rect x={sweep - 1} y={156} width={W} height={10} fill="#e8ecf5" fillOpacity={0.5} />
      {sig.map((_, i) => <line key={i} x1={20 + i * W} x2={20 + i * W} y1={156} y2={166} stroke="#0b0d12" />)}
      <text x={200} y={182} textAnchor="middle" className="tag">pixels</text>
      <text x={200} y={206} textAnchor="middle" className={`verdict-text ${resolved ? 'good' : 'bad'}`}>
        {resolved ? `Resolved — the signal dips to ${(dip * 100).toFixed(0)} % between the lines` : dip >= 0.999 ? 'Not resolved — one merged peak' : `Not resolved — dips only to ${(dip * 100).toFixed(0)} %`}
      </text>
      <text x={200} y={226} textAnchor="middle" className="caption">Each line's image is as wide as the slit's image on the sensor.</text>
    </>
  )
}

/* ── 4. Blaze: tilt the facets to steer light ───────────────────────────── */

function BlazeScene(t: number, wl: number, blaze: number) {
  const eff = blazeEfficiency(wl, blaze, 1)
  const theta = 14 // facet tilt (schematic)
  const y0 = 120
  const facets = Array.from({ length: 7 }, (_, i) => 60 + i * 40)
  const color = wavelengthRgb(wl)
  // Incoming beam from upper left; specular reflection off the tilted facet.
  const inDir: [number, number] = [Math.sin(rad(30)), Math.cos(rad(30))]
  const nrm: [number, number] = [-Math.sin(rad(theta)), -Math.cos(rad(theta))]
  const d = inDir[0] * nrm[0] + inDir[1] * nrm[1]
  const outDir: [number, number] = [inDir[0] - 2 * d * nrm[0], inDir[1] - 2 * d * nrm[1]]
  const dash = (t * 30) % 12
  // Efficiency curve (bottom).
  const lo = blaze * 0.5
  const hi = blaze * 2.2
  const X = (w: number) => 30 + ((w - lo) / (hi - lo)) * 340
  const Y = (e: number) => 228 - e * 50
  const curve = Array.from({ length: 80 }, (_, i) => lo + ((hi - lo) * i) / 79)
    .map((w, i) => `${i ? 'L' : 'M'}${X(w).toFixed(1)} ${Y(blazeEfficiency(w, blaze, 1)).toFixed(1)}`)
    .join('')
  return (
    <>
      {facets.map((x) => (
        <g key={x}>
          <path d={`M${x} ${y0} L${x + 36} ${y0 - 36 * Math.tan(rad(theta))} L${x + 40} ${y0} Z`} fill="#d38bff" fillOpacity={0.35} stroke="#d38bff" />
          <line x1={x + 18 - inDir[0] * 90} y1={y0 - 9 * Math.tan(rad(theta)) * 2 - inDir[1] * 90} x2={x + 18} y2={y0 - 18 * Math.tan(rad(theta))} stroke="#e8ecf5" strokeOpacity={0.5} strokeDasharray="6 6" strokeDashoffset={-dash} />
          <line x1={x + 18} y1={y0 - 18 * Math.tan(rad(theta))} x2={x + 18 + outDir[0] * 90} y2={y0 - 18 * Math.tan(rad(theta)) + outDir[1] * 90} stroke={color} strokeOpacity={0.15 + 0.85 * eff} strokeWidth={1 + 2 * eff} strokeDasharray="6 6" strokeDashoffset={-dash} />
        </g>
      ))}
      <text x={200} y={142} textAnchor="middle" className="tag">facets tilted at the blaze angle mirror the light into order 1</text>
      <rect x={30} y={170} width={340} height={58} fill="#0f1218" />
      <path d={curve} stroke="#d38bff" fill="none" strokeWidth={1.5} />
      <line x1={X(wl)} x2={X(wl)} y1={172} y2={228} stroke={color} strokeWidth={2} />
      <circle cx={X(wl)} cy={Y(eff)} r={3.5} fill={color} />
      <text x={X(wl) + 6} y={Y(eff) - 6} className="tag" fill={color}>{(eff * 100).toFixed(0)} %</text>
      <text x={X(blaze)} y={166} textAnchor="middle" className="tag">blaze {blaze} nm</text>
    </>
  )
}

/* ── 5. Order overlap ───────────────────────────────────────────────────── */

function OrderScene(t: number, filter: boolean) {
  const G = 300
  const alpha = 10
  const gx = 60
  const gy = 120
  const sensorX = 290
  // Map β linearly onto the sensor: 380 nm at the top, 820 nm at the bottom (schematic, exaggerated).
  const b0 = diffractionAngle(380, G, alpha, 1)
  const b1 = diffractionAngle(820, G, alpha, 1)
  const yAt = (wl: number, m = 1) => 52 + ((diffractionAngle(wl, G, alpha, m) - b0) / (b1 - b0)) * 136
  const rays: { wl: number; m: number }[] = []
  for (let wl = 400; wl <= 800; wl += 50) rays.push({ wl, m: 1 })
  rays.push({ wl: 400, m: 2 }) // drawn last: on top of 800 nm, where it lands
  const pulse = (t * 0.6) % 1
  // Order-sorting filter: a long-pass strip over the long-wavelength end of the sensor (≥ ~600 nm).
  const fTop = yAt(590)
  const fBot = yAt(820)
  const fX = sensorX - 9
  return (
    <>
      <rect x={gx - 3} y={gy - 40} width={6} height={80} fill="#d38bff" fillOpacity={0.35} stroke="#d38bff" />
      <line x1={8} y1={gy} x2={gx - 3} y2={gy} stroke="#e8ecf5" strokeOpacity={0.7} strokeWidth={2} />
      {rays.map(({ wl, m }) => {
        const ty = yAt(wl, m)
        const blocked = filter && m === 2
        const ex = blocked ? fX : sensorX
        const ey = gy + ((ty - gy) * (ex - gx)) / (sensorX - gx)
        const color = m === 2 ? '#b48cff' : wavelengthRgb(wl)
        return (
          <g key={`${wl}-${m}`}>
            <line x1={gx} y1={gy} x2={ex} y2={ey} stroke={color} strokeWidth={m === 2 ? 2.2 : 1.3} strokeDasharray={m === 2 ? '5 4' : undefined} strokeOpacity={0.9} />
            <circle cx={gx + (ex - gx) * pulse} cy={gy + (ey - gy) * pulse} r={2} fill={color} />
            {blocked && <circle cx={ex} cy={ey} r={4 + 2 * Math.sin(t * 8)} fill="none" stroke="#b48cff" strokeOpacity={0.7} />}
            {m === 1 && (wl === 400 || wl === 600 || wl === 800) && <text x={sensorX + 14} y={ty + 3} className="tag">{wl} nm</text>}
            {m === 2 && <text x={sensorX + 14} y={ty + 13} className="tag" fill="#b48cff">{blocked ? '400 nm ×2 blocked' : '+ 400 nm, order 2'}</text>}
          </g>
        )
      })}
      <rect x={sensorX} y={46} width={8} height={148} fill="#1b2030" stroke="#4fd6e6" strokeOpacity={0.6} />
      {filter && (
        <>
          <rect x={fX - 3} y={fTop} width={6} height={fBot - fTop} rx={2} fill="#ff6b6b" fillOpacity={0.4} stroke="#ff8a5c" />
          <text x={fX} y={fBot + 12} textAnchor="middle" className="tag">filter</text>
        </>
      )}
      <text x={200} y={228} textAnchor="middle" className="caption">
        {filter
          ? 'A long-pass strip over the red end stops the order-2 blue; order-1 blue still lands.'
          : 'Order 2 of 400 nm leaves at the same angle as order 1 of 800 nm: same pixels.'}
      </text>
    </>
  )
}

export default function Learn() {
  const [wl, setWl] = useState(550)
  const [G, setG] = useState(600)
  const [sep, setSep] = useState(0.6)
  const [width, setWidth] = useState(0.5)
  const [bwl, setBwl] = useState(600)
  const [filter, setFilter] = useState(false)
  return (
    <section className="learn-films">
      <header className="card-head">
        <div>
          <h2>How a spectrometer works</h2>
          <p className="muted">Five short films. They play when scrolled into view; the sliders change the physics.</p>
        </div>
      </header>
      <Chapter n={1} title="The journey of light" lede={<p>Follow white light from the slit to the sensor. Watch which part is doing the work at each moment.</p>}>
        {(t) => Journey(t)}
      </Chapter>
      <Chapter
        n={2}
        title="Inside the grating"
        lede={
          <p>
            Every groove re-emits the light as a small circular wavelet. Only in certain directions do the wavelets line up crest-to-crest — that
            direction, <i>β</i>, depends on the colour. This is the grating equation, <i>mλG = sin α + sin β</i>.
          </p>
        }
        controls={
          <>
            <Slider label="Wavelength" value={wl} min={380} max={780} step={5} unit="nm" onChange={setWl} color={wavelengthRgb(wl)} />
            <Slider label="Grooves" value={G} min={150} max={1200} step={50} unit="/mm" onChange={setG} />
          </>
        }
      >
        {(t) => GratingScene(t, wl, G)}
      </Chapter>
      <Chapter
        n={3}
        title="Resolution: when two lines merge"
        lede={<p>The sodium doublet (589.0 / 589.6 nm) on a row of pixels. Narrow the slit or spread the lines until the dip between them falls below 81 % — the Rayleigh criterion.</p>}
        controls={
          <>
            <Slider label="Line separation" value={sep} min={0.1} max={3} step={0.1} unit="nm" onChange={setSep} />
            <Slider label="Slit image width" value={width} min={0.1} max={2} step={0.1} unit="nm" onChange={setWidth} />
          </>
        }
      >
        {(t) => ResolutionScene(t, sep, width)}
      </Chapter>
      <Chapter
        n={4}
        title="Blaze: steering light into one order"
        lede={<p>Sawtooth facets act as tiny tilted mirrors. Where the mirror direction matches the diffraction direction, that colour is bright; away from the blaze it fades.</p>}
        controls={<Slider label="Wavelength" value={bwl} min={300} max={1300} step={10} unit="nm" onChange={setBwl} color={wavelengthRgb(bwl)} />}
      >
        {(t) => BlazeScene(t, bwl, 500)}
      </Chapter>
      <Chapter
        n={5}
        title="Order overlap"
        lede={<p>A grating also diffracts in second order, at twice the angle-equivalent wavelength. If your span reaches past twice its start, two colours share the same pixels.</p>}
        controls={
          <label className="ctl check">
            <input type="checkbox" checked={filter} onChange={(e) => setFilter(e.target.checked)} /> Order-sorting filter on the sensor
          </label>
        }
      >
        {(t) => OrderScene(t, filter)}
      </Chapter>
    </section>
  )
}

