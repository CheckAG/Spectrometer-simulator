# Grating options
GROOVE_OPTIONS = [300, 600, 1200, 1800, 2400]
DIFFRACTION_ORDER = 1

# Apertures (mm)
LENS_APERTURE = 25.4       # 1 inch
GRATING_APERTURE = 50.8    # 2 inch

# Detector / lens defaults
DEFAULT_PIXEL_SIZE = 5.0   # µm
DEFAULT_F1 = 50.0          # mm
DEFAULT_F2 = 50.0          # mm
DEFAULT_N_PIXELS = 2048
DEFAULT_SAMPLING = 3.0     # pixels per resolution element

# Widget limits
SPECTRAL_RES_MIN = 0.1
SPECTRAL_RES_MAX = 100.0
WAVELENGTH_MIN = 100.0
WAVELENGTH_MAX = 2000.0
PIXEL_SIZE_MIN = 1.0
PIXEL_SIZE_MAX = 100.0
FOCAL_LENGTH_MIN = 0.1
FOCAL_LENGTH_MAX = 200.0
N_PIXELS_MIN = 16
N_PIXELS_MAX = 16384
INCLUDED_ANGLE_MIN = 0.0
INCLUDED_ANGLE_MAX = 90.0

# Defaults
DEFAULT_SPAN_START = 400.0
DEFAULT_SPAN_END = 700.0
DEFAULT_SPECTRAL_RES = 1.0

# Grating types
GRATING_TYPE_TRANSMISSIVE = "Transmissive"
GRATING_TYPE_REFLECTIVE = "Reflective"
GRATING_TYPES = [GRATING_TYPE_TRANSMISSIVE, GRATING_TYPE_REFLECTIVE]
DEFAULT_GRATING_TYPE = GRATING_TYPE_TRANSMISSIVE
# Included angle α − β is computed by default (optics.auto_included_angle):
# reflective = smallest angle at which M1 and M2 clear + this margin (deg).
AUTO_ANGLE_MARGIN = 5.0

# Simulation
N_WAVELENGTH_SAMPLES = 10
N_RAYS = 20
N_SOURCE_POINTS = 2
# Orders drawn faintly besides the design order (0 = undiffracted stray light,
# 2 = order overlap).
EXTRA_ORDERS = (0, 2)
