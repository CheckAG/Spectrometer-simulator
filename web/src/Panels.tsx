/** The requirement sentence, diagnostics, tuning drawer and datasheet. */

import { type ReactNode, useEffect, useState } from 'react'
import { ELEMENT_COLOR } from './color'
import { GROOVE_OPTIONS } from './physics/config'
import { type Design, type Issue, type Spec, blazeEfficiency, defaultBlazeWavelength, designOk, sensorMm } from './physics/optics'
import { type Inputs, LIMITS, clamp } from './state'

type Lim = keyof typeof LIMITS

/** A number typed in place inside running text; commits on blur / Enter. */
export function Token({ value, limit, onChange, label, digits = 0 }: { value: number; limit: Lim; onChange: (v: number) => void; label: string; digits?: number }) {
  const show = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(digits || 2).replace(/0+$/, '').replace(/\.$/, ''))
  const [text, setText] = useState(show(value))
  useEffect(() => setText(show(value)), [value])
  const commit = () => {
    const v = Number(text)
    if (text.trim() && Number.isFinite(v)) onChange(clamp(limit, v))
    else setText(show(value))
  }
  return (
    <input
      className="token"
      aria-label={label}
      inputMode="decimal"
      value={text}
      size={Math.max(text.length, 2)}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault()
          const step = limit === 'resolution' ? 0.1 : limit === 'n_pixels' ? 256 : limit === 'pixel_um' ? 0.5 : 10
          onChange(clamp(limit, Math.round((value + (e.key === 'ArrowUp' ? step : -step)) * 100) / 100))
        }
      }}
    />
  )
}

export function Sentence({ s, set }: { s: Inputs; set: (patch: Partial<Inputs>) => void }) {
  const num = (k: Lim) => (v: number) => set({ [k]: v })
  return (
    <p className="sentence">
      Resolve <Token label="target resolution, nm" value={s.resolution} limit="resolution" onChange={num('resolution')} digits={2} /> nm
      across <Token label="shortest wavelength, nm" value={s.wl_min} limit="wl_min" onChange={num('wl_min')} />–
      <Token label="longest wavelength, nm" value={s.wl_max} limit="wl_max" onChange={num('wl_max')} /> nm on a{' '}
      <Token label="pixel count" value={s.n_pixels} limit="n_pixels" onChange={num('n_pixels')} /> ×{' '}
      <Token label="pixel pitch, µm" value={s.pixel_um} limit="pixel_um" onChange={num('pixel_um')} digits={1} /> µm sensor, with a{' '}
      <span className="switch" role="radiogroup" aria-label="Layout">
        <button role="radio" aria-checked={s.transmissive} onClick={() => set({ transmissive: true, angle: null })}>
          lens (4f)
        </button>
        <button role="radio" aria-checked={!s.transmissive} onClick={() => set({ transmissive: false, angle: null })}>
          mirror (Czerny–Turner)
        </button>
      </span>{' '}
      spectrometer.
    </p>
  )
}

/** Which bench elements a warning is about (0 collimator, 1 grating, 2 focuser, 3 sensor). */
export function issueParts(w: Issue): number[] {
  if (w.text.startsWith('M1 and M2') || w.text.startsWith('Mirror aberrations')) return [0, 2]
  return ({ f2: [2, 3], n_pixels: [3], resolution: [3], included_deg: [1], wl_max: [1], blaze_nm: [1] } as Record<string, number[]>)[w.field ?? ''] ?? []
}

export function Diagnostics({ design, onFix, onHover }: { design: Design; onFix: (fix: Partial<Spec>) => void; onHover: (parts: number[]) => void }) {
  if (!design.warnings.length) {
    return (
      <section className="card diagnostics ok">
        <h2>Diagnostics</h2>
        <p className="all-clear">
          <span className="dot good" /> Meets the target and fits the sensor.
        </p>
      </section>
    )
  }
  return (
    <section className="card diagnostics">
      <h2>
        Diagnostics <span className="count">{design.warnings.length}</span>
      </h2>
      {design.warnings.map((w) => {
        const [head, ...rest] = w.text.split(/(?<=\.) /)
        return (
          <article key={w.text} className="issue" onPointerEnter={() => onHover(issueParts(w))} onPointerLeave={() => onHover([])}>
            <p>
              <b>{head}</b> {rest.join(' ')}
            </p>
            {Object.keys(w.fixes).length > 0 && (
              <div className="fixes">
                {Object.entries(w.fixes).map(([label, fix]) => (
                  <button key={label} onClick={() => onFix(fix)}>
                    {label}
                  </button>
                ))}
              </div>
            )}
          </article>
        )
      })}
    </section>
  )
}

function Row({ k, v, unit, note }: { k: string; v: ReactNode; unit?: string; note?: string }) {
  return (
    <div className="ds-row" title={note}>
      <dt>{k}</dt>
      <dd>
        {v}
        {unit && <span className="unit"> {unit}</span>}
      </dd>
    </div>
  )
}

const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—')

/** Relative blaze efficiency across the span, as a tiny inline chart. */
function EfficiencySpark({ spec, design }: { spec: Spec; design: Design }) {
  const pad = 0.3 * (spec.wl_max - spec.wl_min)
  const lo = Math.max(spec.wl_min - pad, 1)
  const hi = spec.wl_max + pad
  const x = (wl: number) => ((wl - lo) / (hi - lo)) * 200
  const pts = Array.from({ length: 81 }, (_, i) => lo + ((hi - lo) * i) / 80)
  const path = pts.map((wl, i) => `${i ? 'L' : 'M'}${x(wl).toFixed(1)} ${(46 - blazeEfficiency(wl, design.blaze_nm, spec.m) * 42).toFixed(1)}`).join('')
  return (
    <svg className="spark" viewBox="0 0 200 50" aria-label="Blaze efficiency across the span">
      <rect x={x(spec.wl_min)} width={x(spec.wl_max) - x(spec.wl_min)} y={0} height={50} className="span" />
      <path d={path} stroke={ELEMENT_COLOR.grating} fill="none" strokeWidth={1.6} />
      <line x1={x(design.blaze_nm)} x2={x(design.blaze_nm)} y1={0} y2={50} stroke={ELEMENT_COLOR.grating} strokeDasharray="2 2" />
    </svg>
  )
}

export function Datasheet({ spec, design, G }: { spec: Spec; design: Design; G: number }) {
  const sensor = sensorMm(spec)
  const fill = design.spectrum_mm / sensor
  const ok = designOk(design)
  return (
    <section className="card datasheet">
      <header className="card-head">
        <div>
          <h2>Datasheet</h2>
          <p className="muted">{spec.transmissive ? '4f transmission spectrometer' : 'Czerny–Turner spectrometer'}</p>
        </div>
      </header>
      <div className="ds-grid">
        <dl>
          <h3>Performance</h3>
          <Row k="Resolution" v={fmt(design.resolution, 3)} unit="nm" />
          <Row k="  slit / pixel limit" v={fmt(design.resolution_image, 3)} unit="nm" />
          <Row k="  grating limit" v={fmt(design.resolution_grating, 3)} unit="nm" />
          {design.aberration_nm.length > 0 && <Row k="  mirror aberrations" v={`${fmt(Math.min(...design.aberration_nm), 2)}–${fmt(Math.max(...design.aberration_nm), 2)}`} unit="nm" />}
          <Row k="Dispersion (reciprocal)" v={fmt(design.rld, 2)} unit="nm/mm" />
          <Row k="Spectrum length" v={`${fmt(design.spectrum_mm, 2)} of ${sensor.toFixed(2)}`} unit="mm" />
          <div className="fill" aria-label={`Sensor ${fmt(fill * 100, 0)} % used`}>
            <span style={{ width: `${Math.min(fill, 1) * 100}%` }} className={fill > 1 ? 'over' : ''} />
          </div>
        </dl>
        <dl>
          <h3>Grating</h3>
          <Row k="Grooves" v={G} unit="l/mm" />
          <Row k="Blaze" v={`${fmt(design.blaze_nm, 0)} nm · ${fmt(design.blaze_angle, 1)}°`} />
          <EfficiencySpark spec={spec} design={design} />
          <Row k="Included angle 2K" v={fmt(design.included_deg, 0)} unit="°" />
          <Row k="Incidence α" v={fmt(design.alpha, 2)} unit="°" />
          <Row k={`β ${spec.wl_min}–${spec.wl_max} nm`} v={`${fmt(design.beta_min, 1)} … ${fmt(design.beta_max, 1)}`} unit="°" />
        </dl>
        <dl>
          <h3>Optics</h3>
          <Row k="Slit width" v={fmt(design.slit_um, 1)} unit="µm" />
          <Row k="f1 · f2" v={`${spec.f1} · ${spec.f2}`} unit="mm" />
          <Row k="Beam" v={`${spec.beam_mm} mm · f/${(spec.f1 / spec.beam_mm).toFixed(1)}`} />
          {!spec.transmissive && <Row k="Mirror off-axis M1 · M2" v={`${fmt(design.fold1, 1)} · ${fmt(design.fold2, 1)}`} unit="°" />}
          {!ok && <p className="verdict bad">No valid geometry for this grating.</p>}
        </dl>
      </div>
    </section>
  )
}

export function Tune({ s, set, design, recommendedG, autoAngle }: { s: Inputs; set: (p: Partial<Inputs>) => void; design: Design; recommendedG: number; autoAngle: number }) {
  const blazeAuto = defaultBlazeWavelength(s.wl_min, s.wl_max)
  return (
    <section className="card tune">
      <h2>Fine-tune</h2>
      <label>
        <span>Grating</span>
        <select value={design.G} onChange={(e) => set({ G: Number(e.target.value) })}>
          {GROOVE_OPTIONS.map((g) => (
            <option key={g} value={g}>
              {g} l/mm{g === recommendedG ? ' ★ recommended' : ''}
            </option>
          ))}
        </select>
        {s.G !== null && s.G !== recommendedG && <button className="link" onClick={() => set({ G: null })}>use recommended</button>}
      </label>
      <label>
        <span>Included angle <b>{design.included_deg.toFixed(0)}°</b></span>
        <input type="range" min={0} max={90} value={design.included_deg} onChange={(e) => set({ angle: Number(e.target.value) })} />
        {s.angle !== null ? <button className="link" onClick={() => set({ angle: null })}>use computed ({autoAngle}°)</button> : <small className="muted">computed</small>}
      </label>
      <label>
        <span>Pixels per resolution element <b>{s.sampling}</b></span>
        <input type="range" min={1} max={10} step={0.5} value={s.sampling} onChange={(e) => set({ sampling: Number(e.target.value) })} />
      </label>
      <label>
        <span>Beam diameter <b>{s.beam_mm} mm</b></span>
        <input type="range" min={LIMITS.beam_mm[0]} max={LIMITS.beam_mm[1]} step={0.1} value={s.beam_mm} onChange={(e) => set({ beam_mm: Number(e.target.value) })} />
      </label>
      <label>
        <span>Blaze wavelength <b>{(s.blaze ?? blazeAuto).toFixed(0)} nm</b></span>
        <input type="range" min={Math.max(100, Math.round(s.wl_min * 0.5))} max={Math.round(s.wl_max * 1.5)} value={Math.round(s.blaze ?? blazeAuto)} onChange={(e) => set({ blaze: Number(e.target.value) })} />
        {s.blaze !== null && <button className="link" onClick={() => set({ blaze: null })}>use computed ({blazeAuto.toFixed(0)} nm)</button>}
      </label>
      <label className="check" title={s.transmissive ? 'Czerny–Turner only' : 'Trace real spherical mirrors: spherical aberration, coma, off-axis focus shift'}>
        <input type="checkbox" checked={s.exact} disabled={s.transmissive} onChange={(e) => set({ exact: e.target.checked })} />
        Real mirrors (trace aberrations){s.transmissive && <small className="muted"> · mirror layout only</small>}
      </label>
    </section>
  )
}
