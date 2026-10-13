"""DT source model used for the validation (density in m^-3, temperature in keV)."""
import numpy as np


def plasma_conditions(s):
    return 4.8e20 * (1 - s**5), 11.5 * (1 - s)


def reaction_rate(n_i, T_i):
    """Bosch and Hale (1992), Nucl. Fusion 32, 611; reactions cm^-3 s^-1."""
    if T_i == 0 or n_i == 0:
        return 0.0
    c1, c2, c3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
    c4, c5, c6, c7 = 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5
    theta = T_i / (1 - T_i * (c2 + T_i * (c4 + T_i * c6)) / (1 + T_i * (c3 + T_i * (c5 + T_i * c7))))
    xi = (34.3827**2 / (4 * theta))**(1 / 3)
    sigma_v = c1 * theta * np.sqrt(xi / (1124656 * T_i**3)) * np.exp(-3 * xi)
    return ((n_i / 1e6)**2 / 4) * sigma_v
