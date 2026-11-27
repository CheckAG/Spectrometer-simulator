// Python-style number formatting, so warning texts match the Python core.

/** Python `format(x, '.{p}g')`. */
export function g(x: number, p = 6): string {
  if (Number.isNaN(x)) return 'nan'
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf'
  if (x === 0) return '0'
  const [mant, ex] = x.toExponential(p - 1).split('e')
  const e = Number(ex)
  const strip = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s)
  if (e < -4 || e >= p) return `${strip(mant)}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`
  return strip(x.toFixed(Math.max(p - 1 - e, 0)))
}

/** Python `format(x, '.{d}f')`. */
export const f = (x: number, d: number) => x.toFixed(d)

/** Python `format(x, '.0%')`. */
export const pct = (x: number) => `${(x * 100).toFixed(0)}%`
