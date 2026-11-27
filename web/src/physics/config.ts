// Port of spectrometer/config.py.
export const GROOVE_OPTIONS = [300, 600, 1200, 1800, 2400]
export const DIFFRACTION_ORDER = 1

export const LENS_APERTURE = 25.4 // mm, 1 inch
export const GRATING_APERTURE = 50.8 // mm, 2 inch

export const DEFAULT_PIXEL_SIZE = 5.0
export const DEFAULT_F1 = 50.0
export const DEFAULT_F2 = 50.0
export const DEFAULT_N_PIXELS = 2048
export const DEFAULT_SAMPLING = 3.0

export const SPECTRAL_RES_MIN = 0.1
export const SPECTRAL_RES_MAX = 100.0
export const WAVELENGTH_MIN = 100.0
export const WAVELENGTH_MAX = 2000.0
export const PIXEL_SIZE_MIN = 1.0
export const PIXEL_SIZE_MAX = 100.0
export const FOCAL_LENGTH_MIN = 0.1
export const FOCAL_LENGTH_MAX = 200.0
export const N_PIXELS_MIN = 16
export const N_PIXELS_MAX = 16384
export const INCLUDED_ANGLE_MIN = 0.0
export const INCLUDED_ANGLE_MAX = 90.0

export const DEFAULT_SPAN_START = 400.0
export const DEFAULT_SPAN_END = 700.0
export const DEFAULT_SPECTRAL_RES = 1.0

// Reflective auto angle = smallest angle at which M1 and M2 clear + this margin (deg).
export const AUTO_ANGLE_MARGIN = 5.0

export const N_WAVELENGTH_SAMPLES = 10
export const N_RAYS = 20
export const N_SOURCE_POINTS = 2
export const EXTRA_ORDERS = [0, 2]
