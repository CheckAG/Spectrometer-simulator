import { useEffect, useMemo, useState } from 'react'
import Bench from './Bench'
import Learn from './Learn'
import { Datasheet, Diagnostics, Sentence, Token, Tune, issueParts } from './Panels'
import SensorView from './SensorView'
import { analyze, designOk, recommendGrooveDensity, sensorMm } from './physics/optics'
import { type Inputs, applyFix, clamp, fromQuery, toQuery, toSpec } from './state'

// Streamlit version; it reads the same query keys, so links carry the design across.
const CLASSIC_APP = 'https://checkag-spectral-simulator.streamlit.app/'

export default function App() {
  const [s, setS] = useState<Inputs>(() => fromQuery(window.location.search))
  const [stray, setStray] = useState(false)
  const [hovered, setHovered] = useState<number[]>([])
  const [copied, setCopied] = useState(false)
  const set = (patch: Partial<Inputs>) => setS((o) => ({ ...o, ...patch }))

  useEffect(() => window.history.replaceState(null, '', `?${toQuery(s)}`), [s])

  const valid = s.wl_min < s.wl_max
  const spec = useMemo(() => toSpec(s), [s])
  const result = useMemo(() => {
    if (!valid) return null
    const recommended = recommendGrooveDensity(spec)
    const G = s.G ?? recommended.G
    const design = analyze(spec, G)
    const autoAngle = analyze({ ...spec, included_deg: null, exact_mirrors: false }, G, false).included_deg
    return { recommended, G, design, autoAngle }
  }, [spec, s.G, valid])

  const design = result?.design
  const ok = design ? designOk(design) : false
  const flagged = useMemo(() => {
    if (hovered.length) return new Set(hovered)
    return new Set(design?.warnings.flatMap(issueParts) ?? [])
  }, [hovered, design])

  const exportJson = () => {
    if (!design) return
    const { warnings, ...values } = design
    const blob = new Blob([JSON.stringify({ spec, design: values, warnings: warnings.map((w) => w.text) }, null, 2)], { type: 'application/json' })
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'spectrometer-design.json' })
    a.click()
    URL.revokeObjectURL(a.href)
  }
  const copyLink = () =>
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })

  const res = design?.resolution ?? NaN
  const meets = res <= s.resolution
  const fill = design ? design.spectrum_mm / sensorMm(spec) : NaN

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" aria-hidden className="logo">
            <path d="M4 16 L14 16" stroke="currentColor" strokeWidth="2" />
            <path d="M14 10 L14 22" stroke="currentColor" strokeWidth="2.5" />
            <path d="M14 16 L28 8" stroke="#7fb2ff" strokeWidth="2" />
            <path d="M14 16 L28 16" stroke="#5fd39a" strokeWidth="2" />
            <path d="M14 16 L28 24" stroke="#ff6b6b" strokeWidth="2" />
          </svg>
          <span>Spectrometer Designer</span>
        </div>
        <nav className="actions">
          <button onClick={copyLink}>{copied ? 'Link copied ✓' : 'Copy link'}</button>
          <button onClick={exportJson} disabled={!design}>Export JSON</button>
          <a href={`${CLASSIC_APP}?${toQuery(s)}`} target="_blank" rel="noreferrer" title="The Streamlit version, opened on this same design">
            Classic app
          </a>
          <a href="https://github.com/CheckAG/Spectrometer-simulator" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </header>

      <Sentence s={s} set={set} />
      {!valid && <p className="verdict bad">The span must end at a longer wavelength than it starts.</p>}

      {result && design && (
        <>
          <section className="bench-card">
            <div className="hud">
              <div className={`hud-main ${ok && meets ? 'good' : 'bad'}`}>
                <span className="hud-label">resolution</span>
                <span className="hud-value">
                  {ok ? res.toFixed(3) : '—'}
                  <small> nm</small>
                </span>
                <span className="hud-sub">
                  {ok ? (meets ? `meets the ${s.resolution} nm target` : `misses the ${s.resolution} nm target`) : 'no valid geometry'}
                  {design.aberration_nm.length ? ' · real mirrors' : ''}
                </span>
              </div>
              <div className="hud-stat">
                <span className="hud-label">sensor fill</span>
                <span className="hud-value small">{Number.isFinite(fill) ? `${(fill * 100).toFixed(0)} %` : '—'}</span>
                <span className="meter">
                  <span style={{ width: `${Math.min(fill, 1) * 100}%` }} className={fill > 1 ? 'over' : ''} />
                </span>
              </div>
              <div className="hud-stat">
                <span className="hud-label">grating</span>
                <span className="hud-value small">
                  {result.G}
                  <small> l/mm</small>
                </span>
                <span className="hud-sub">{s.G === null ? 'recommended' : 'manual'}</span>
              </div>
              <div className="hud-controls">
                <label>
                  f1 <Token label="f1, mm" value={s.f1} limit="f1" onChange={(v) => set({ f1: v })} /> mm
                </label>
                <label>
                  f2 <Token label="f2, mm" value={s.f2} limit="f2" onChange={(v) => set({ f2: v })} /> mm
                </label>
                <label className="check">
                  <input type="checkbox" checked={stray} onChange={(e) => setStray(e.target.checked)} /> stray light
                </label>
              </div>
            </div>
            {ok ? (
              <Bench
                spec={spec}
                design={design}
                exact={design.aberration_nm.length > 0}
                stray={stray}
                highlight={flagged}
                onF1={(v) => set({ f1: clamp('f1', v) })}
                onF2={(v) => set({ f2: clamp('f2', v) })}
                onAngle={(v) => set({ angle: clamp('angle', v) })}
              />
            ) : (
              <div className="bench-empty">This grating can't diffract the whole span at this angle — see Diagnostics.</div>
            )}
            <p className="bench-hint">
              Drag the rings on {spec.transmissive ? 'F1 / F2' : 'M1 / M2'} along the beam to change focal lengths · drag the grating dial sideways to turn it
            </p>
          </section>

          <div className="grid-2">
            {ok ? <SensorView spec={spec} design={design} /> : <section className="card" />}
            <Diagnostics design={design} onFix={(fix) => setS((o) => applyFix(o, fix))} onHover={setHovered} />
          </div>

          <div className="grid-2 wide-left">
            <Datasheet spec={spec} design={design} G={result.G} />
            <Tune s={s} set={set} design={design} recommendedG={result.recommended.G} autoAngle={result.autoAngle} />
          </div>

          <Learn />
        </>
      )}

      <footer className="foot muted">Physics runs entirely in your browser · ported from the Python core and tested against it.</footer>
    </div>
  )
}
