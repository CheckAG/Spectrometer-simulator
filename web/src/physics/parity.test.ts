// The TS physics must reproduce the Python core (spectrometer/). Fixtures come
// from web/scripts/make_fixtures.py; regenerate them after changing the Python side.
import { describe, expect, test } from 'vitest'
import fixtures from './fixtures.json'
import { sensorHits, trace } from './layout'
import { type Spec, analyze, detectorSignal, lineDip, recommendGrooveDensity } from './optics'

type Num = number | null

/** Equal within a relative/absolute tolerance; null (NaN) only matches NaN. */
function close(actual: number, expected: Num, rel = 1e-9, abs = 1e-9) {
  if (expected === null) return expect(actual, 'expected NaN').toBeNaN()
  expect(Math.abs(actual - expected), `${actual} vs ${expected}`).toBeLessThanOrEqual(abs + rel * Math.abs(expected))
}

describe.each(fixtures)('$name', (fx) => {
  const spec = fx.spec as unknown as Spec
  const design = analyze(spec, fx.G)

  test('recommended groove density', () => {
    expect(recommendGrooveDensity(spec).G).toBe(fx.recommended)
  })

  test('design values', () => {
    for (const [key, expected] of Object.entries(fx.design)) {
      const actual = design[key as keyof typeof design]
      if (Array.isArray(expected)) {
        expect((actual as number[]).length, key).toBe(expected.length)
        // Traced blur: float summation order differs slightly from numpy.
        expected.forEach((e, i) => close((actual as number[])[i], e as Num, 1e-6, 1e-9))
      } else if (typeof expected === 'boolean') {
        expect(actual, key).toBe(expected)
      } else {
        close(actual as number, expected as Num, key.includes('aberration') || key === 'resolution' ? 1e-6 : 1e-9)
      }
    }
  })

  test('warnings and fixes', () => {
    expect(design.warnings.map((w) => w.field)).toEqual(fx.warnings.map((w) => w.field))
    expect(design.warnings.map((w) => w.text)).toEqual(fx.warnings.map((w) => w.text))
    expect(design.warnings.map((w) => w.fixes)).toEqual(fx.warnings.map((w) => w.fixes))
  })

  test('detector read-out', () => {
    const [wlPix, signal] = detectorSignal(spec, design, fx.detector.lines)
    const step = fx.detector.step
    fx.detector.wl_pix.forEach((w, i) => close(wlPix[i * step], w as Num, 1e-7, 1e-6))
    fx.detector.signal.forEach((s, i) => close(signal[i * step], s as Num, 1e-6, 1e-9))
    close(lineDip(wlPix, signal, 576.96, 579.07), fx.detector.dip_577 as Num, 1e-6, 1e-9)
  })

  test('traced sensor hits', () => {
    const wls = fx.trace.wavelengths
    for (const mode of ['ideal', 'exact'] as const) {
      const ref = (fx.trace as Record<string, unknown>)[mode] as
        | { mean: Num[]; sensor_pos: number[]; sensor_theta: number }
        | undefined
      if (!ref) continue
      const t = trace(spec, design, wls, undefined, mode === 'exact')
      const sensor = t.elements[t.elements.length - 1]
      close(sensor.pos[0], ref.sensor_pos[0], 1e-7, 1e-7)
      close(sensor.pos[1], ref.sensor_pos[1], 1e-7, 1e-7)
      close(sensor.theta, ref.sensor_theta, 1e-7, 1e-9)
      const hits = sensorHits(t)
      ref.mean.forEach((m, k) => {
        const own = Array.from(hits.slice(k * t.raysPerWl, (k + 1) * t.raysPerWl)).filter(Number.isFinite)
        close(own.length ? own.reduce((a, b) => a + b, 0) / own.length : NaN, m, 1e-7, 1e-7)
      })
    }
  })
})
