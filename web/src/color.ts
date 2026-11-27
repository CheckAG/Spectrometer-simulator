/** Perceived colour of a wavelength (Bruton), as [r, g, b] in 0–1; dim violet/red outside 380–780 nm. */
export function wavelengthRgbTuple(nm: number): [number, number, number] {
  if (nm < 380) return [0.43, 0.16, 0.55]
  if (nm > 780) return [0.55, 0.12, 0.12]
  const bands: [number, number, (t: number) => [number, number, number]][] = [
    [380, 440, (t) => [1 - t, 0, 1]],
    [440, 490, (t) => [0, t, 1]],
    [490, 510, (t) => [0, 1, 1 - t]],
    [510, 580, (t) => [t, 1, 0]],
    [580, 645, (t) => [1, 1 - t, 0]],
    [645, 781, () => [1, 0, 0]],
  ]
  const [lo, hi, f] = bands.find(([, hi]) => nm < hi)!
  const [r, g, b] = f((nm - lo) / (hi - lo))
  const fade = 0.4 + 0.6 * Math.min(1, (nm - 380) / 40, (780 - nm) / 80) // dim toward the edges of vision
  return [r * fade, g * fade, b * fade]
}

export function wavelengthRgb(nm: number, alpha = 1): string {
  const [r, g, b] = wavelengthRgbTuple(nm).map((v) => Math.round(v * 255))
  return alpha === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`
}

export const ELEMENT_COLOR: Record<string, string> = {
  lens: '#7fb2ff',
  spherical_mirror: '#5fd39a',
  grating: '#d38bff',
  sensor: '#4fd6e6',
}
export const SLIT_COLOR = '#f2b84b'
