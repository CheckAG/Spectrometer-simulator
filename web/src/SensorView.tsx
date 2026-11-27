/**
 * What the sensor sees: the simulated read-out rendered as a spectrum strip
 * (each pixel lit in its wavelength's colour) over an intensity trace, with
 * verdicts for close line pairs.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { wavelengthRgb, wavelengthRgbTuple } from './color'
import { type Design, type Spec, HG_AR_LINES, blazeEfficiency, detectorSignal, lineDip } from './physics/optics'

const SOURCES = {
  lamp: { label: 'Hg–Ar lamp', lines: HG_AR_LINES },
  sodium: { label: 'Sodium D', lines: [588.995, 589.592] },
  custom: { label: 'Custom', lines: [] as number[] },
} as const
type Source = keyof typeof SOURCES

export default function SensorView({ spec, design }: { spec: Spec; design: Design }) {
  const [source, setSource] = useState<Source>('lamp')
  const [text, setText] = useState('486.1, 656.3')
  const [hover, setHover] = useState<number | null>(null)
  const canvas = useRef<HTMLCanvasElement>(null)

  const custom = text.split(/[,;\s]+/).filter(Boolean).map(Number)
  const bad = source === 'custom' && custom.some((v) => !Number.isFinite(v))
  const pool = source === 'custom' ? (bad ? [] : custom) : SOURCES[source].lines
  const lines = pool.filter((l) => l > 0)
  const key = lines.join(',')

  const out = useMemo(() => {
    if (!lines.length) return null
    const [wlPix, signal] = detectorSignal(spec, design, lines, lines.map((l) => blazeEfficiency(l, design.blaze_nm, spec.m)))
    const peak = Math.max(...signal, 1e-12)
    const sorted = [...lines].sort((a, b) => a - b)
    const pairs = sorted
      .slice(1)
      .map((b, i) => [sorted[i], b] as const)
      .filter(([a, b]) => b - a < 3 * design.resolution)
      .map(([a, b]) => ({ a, b, dip: lineDip(wlPix, signal, a, b) }))
    return { wlPix, signal: Float64Array.from(signal, (s) => s / peak), pairs }
  }, [spec, design, key])

  // Paint the strip: one column per pixel, coloured by wavelength, brightness = signal.
  useEffect(() => {
    const cv = canvas.current
    if (!cv) return
    const n = spec.n_pixels
    cv.width = n
    cv.height = 1
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(n, 1)
    for (let i = 0; i < n; i++) {
      const wl = out?.wlPix[i] ?? NaN
      const s = out ? Math.pow(out.signal[i], 0.6) : 0
      const [r, g, b] = Number.isFinite(wl) ? wavelengthRgbTuple(wl) : [0, 0, 0]
      // A faint continuum glow keeps the sensor's wavelength scale visible between lines.
      const base = Number.isFinite(wl) ? 0.06 : 0
      img.data.set([255 * Math.min(1, r * s + base * r), 255 * Math.min(1, g * s + base * g), 255 * Math.min(1, b * s + base * b), 255], i * 4)
    }
    ctx.putImageData(img, 0, 0)
  }, [out, spec.n_pixels])

  const n = spec.n_pixels
  const trace = out
    ? Array.from(out.signal, (s, i) => `${i ? 'L' : 'M'}${((i + 0.5) / n * 1000).toFixed(2)} ${(100 - s * 92).toFixed(2)}`).join('')
    : ''
  const finite = out ? Array.from(out.wlPix).filter(Number.isFinite) : []
  const wlLo = finite.length ? Math.min(...finite) : spec.wl_min
  const wlHi = finite.length ? Math.max(...finite) : spec.wl_max
  // Pixel index of a wavelength (wavelength grows with pixel index or against it).
  const pixelOf = (wl: number) => {
    if (!out) return NaN
    let best = NaN
    let dist = Infinity
    for (let i = 0; i < n; i += 1) {
      const d = Math.abs(out.wlPix[i] - wl)
      if (d < dist) [dist, best] = [d, i]
    }
    return dist < 5 * design.rld * spec.pixel_um * 1e-3 ? best : NaN
  }
  const spanPix = [pixelOf(spec.wl_min), pixelOf(spec.wl_max)].filter(Number.isFinite).sort((a, b) => a - b)
  const hoverWl = hover !== null && out ? out.wlPix[hover] : NaN

  return (
    <section className="card sensor-view">
      <header className="card-head">
        <div>
          <h2>What the sensor sees</h2>
          <p className="muted">
            {n} pixels · {spec.pixel_um} µm · {wlLo.toFixed(0)}–{wlHi.toFixed(0)} nm reach the sensor
          </p>
        </div>
        <div className="chips" role="radiogroup" aria-label="Light source">
          {(Object.keys(SOURCES) as Source[]).map((s) => (
            <button key={s} role="radio" aria-checked={source === s} onClick={() => setSource(s)}>
              {SOURCES[s].label}
            </button>
          ))}
        </div>
      </header>
      {source === 'custom' && (
        <label className="inline-field">
          Lines (nm)
          <input value={text} onChange={(e) => setText(e.target.value)} aria-invalid={bad} />
        </label>
      )}

      <div
        className="strip-wrap"
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          setHover(Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * n))))
        }}
        onPointerLeave={() => setHover(null)}
      >
        <canvas ref={canvas} className="strip" aria-label="Simulated spectrum on the sensor" />
        <svg className="trace" viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden>
          {spanPix.length === 2 && (
            <rect x={(spanPix[0] / n) * 1000} width={((spanPix[1] - spanPix[0]) / n) * 1000} y={0} height={100} className="span" />
          )}
          <path d={trace} className="trace-line" vectorEffect="non-scaling-stroke" />
          {hover !== null && <line x1={((hover + 0.5) / n) * 1000} x2={((hover + 0.5) / n) * 1000} y1={0} y2={100} className="cursor" vectorEffect="non-scaling-stroke" />}
        </svg>
        <div className="readout">
          {hover !== null && out ? (
            <>
              pixel <b>{hover}</b> · <b style={{ color: Number.isFinite(hoverWl) ? wavelengthRgb(hoverWl) : undefined }}>{Number.isFinite(hoverWl) ? `${hoverWl.toFixed(2)} nm` : '—'}</b> · signal <b>{out.signal[hover].toFixed(3)}</b>
            </>
          ) : (
            <span className="muted">Hover the sensor to read pixels</span>
          )}
        </div>
      </div>

      {bad && <p className="verdict bad">Enter wavelengths separated by commas.</p>}
      {!lines.length && !bad && <p className="verdict">No lines to show.</p>}
      <div className="verdicts">
        {out?.pairs.map(({ a, b, dip }) => (
          <p key={`${a}-${b}`} className={`verdict ${Number.isNaN(dip) ? '' : dip <= 0.81 ? 'good' : 'bad'}`}>
            <b>{a} / {b} nm</b> · {(b - a).toFixed(2)} nm apart ·{' '}
            {Number.isNaN(dip)
              ? 'not both on the sensor'
              : dip <= 0.81
                ? `resolved, dips to ${(dip * 100).toFixed(0)} %`
                : dip < 1
                  ? `not resolved, dips only to ${(dip * 100).toFixed(0)} % (Rayleigh ≤ 81 %)`
                  : 'not resolved, one merged peak'}
          </p>
        ))}
      </div>
    </section>
  )
}
