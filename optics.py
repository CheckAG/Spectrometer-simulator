import numpy as np
from config import GROOVE_OPTIONS


def closest_value(input_list: list, input_value: float) -> int:
    arr = np.asarray(input_list)
    return arr[(np.abs(arr - input_value)).argmin()]


def calculate_blaze_wavelength(span_start: float, span_end: float) -> float:
    return (span_start + span_end) / 2.0


def calculate_optimal_N(blaze_wavelength_nm: float, spectral_resolution_nm: float,
                         groove_options: list = GROOVE_OPTIONS) -> int:
    raw = blaze_wavelength_nm / spectral_resolution_nm
    return int(closest_value(groove_options, raw))


def calculate_blaze_angle(blaze_wavelength_nm: float, N: int, m: int = 1) -> float:
    d_mm = 1.0 / N
    arg = (m * blaze_wavelength_nm * 1e-9) / (2.0 * d_mm * 1e-3)
    arg = np.clip(arg, -1.0, 1.0)
    return np.round(np.degrees(np.arcsin(arg)), 2)


def calculate_incident_angle(blaze_wavelength_nm: float, N: int,
                              blaze_angle_deg: float, m: int = 1) -> float:
    blazewl_m_N = blaze_wavelength_nm * 1e-6 * m * N
    sin_blaze = np.sin(np.radians(blaze_angle_deg))
    arg = blazewl_m_N / (2.0 * sin_blaze)
    if np.abs(arg) > 1.0:
        return float("nan")
    return np.round(np.degrees(np.arccos(arg)) + blaze_angle_deg, 2)


def calculate_beta(wavelength_nm: float, N: int,
                   incident_angle_deg: float, m: int = 1) -> float:
    m_lambda_N = m * wavelength_nm * 1e-6 * N
    sin_alpha = np.sin(np.radians(incident_angle_deg))
    arg = np.clip(m_lambda_N - sin_alpha, -1.0, 1.0)
    return np.round(np.degrees(np.arcsin(arg)), 2)


def calculate_optical_dispersion(blaze_wavelength_nm: float, N: int,
                                  incident_angle_deg: float, m: int = 1) -> float:
    blazewl_m_N = blaze_wavelength_nm * 1e-6 * m * N
    sin_alpha = np.sin(np.radians(incident_angle_deg))
    denom = np.sqrt(1.0 - (blazewl_m_N - sin_alpha) ** 2)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return np.round((m * N) / denom, 2)


def calculate_f2_prescribed(pixel_size_um: float, optical_dispersion: float,
                             spectral_resolution_nm: float) -> float:
    numerator = 1.5 * pixel_size_um * 1e-3
    denominator = np.tan(optical_dispersion * spectral_resolution_nm * 1e-6) / 2.0
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return np.round(numerator / denominator, 2)


def calculate_sensor_width(f2_mm: float, beta_min_deg: float,
                            beta_max_deg: float) -> float:
    span_rad = np.radians(beta_max_deg - beta_min_deg)
    return np.round(2.0 * f2_mm * np.tan(span_rad / 2.0), 2)


def calculate_slit_width(f1_mm: float, f2_mm: float, pixel_size_um: float) -> float:
    return np.round((f1_mm * 3.0 * pixel_size_um * 1e-3) / f2_mm, 5)


def calculate_optical_resolution(f2_mm: float, pixel_size_um: float,
                                  optical_dispersion: float) -> float:
    if optical_dispersion == 0 or np.isnan(optical_dispersion):
        return float("nan")
    angle = np.arctan(1.5 * pixel_size_um * 1e-6 / f2_mm)
    return np.round((2.0 * angle / optical_dispersion) * 1e9, 4)
