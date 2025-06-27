from scipy.optimize import curve_fit
import numpy as np
from astropy.constants import c
import astropy.units as u

def cube_fit_Halpha(flux, wave, window_centre=6562.8, ws=40, threshold=0.1, std=0.5):
    _, xrange, yrange = flux.shape
    vel_map = np.full((yrange, xrange), np.nan)
    flux_map = {}
    wave_map = {}
    popt_map = {}

    for y in range(yrange):
        for x in range(xrange):
            try:
                velocity, flux_cut, wave_cut, popt = fit_Halpha(x, y, flux, wave, window_centre=window_centre, ws=ws, threshold=threshold, std=std)
                vel_map[y, x] = velocity
                flux_map[(y, x)] = flux_cut
                wave_map[(y, x)] = wave_cut
                popt_map[(y, x)] = popt
            except:
                continue

    return vel_map, flux_map, wave_map, popt_map

def fit_Halpha(x, y, flux, wave, ws=40, threshold=0.1, window_centre=6562.8, std=0.5):
    peak_flux = wave[flux[:, y, x].argmax()]

    bounds = (
        [0, peak_flux-10, 0.1, -np.inf],  # min values
        [np.inf, peak_flux+10, 3, np.inf]  # max sigma = 3 Å ~ 130 km/s
    )
    
    line_mask = (wave > window_centre - ws) & (wave < window_centre + ws)
    wave_cut = wave[line_mask]
    flux_cut = flux[line_mask, y, x]

    if not has_peak(flux_cut, threshold=threshold): raise ValueError("No significant peak found in the spectrum.")
    if np.isnan(flux_cut).all(): raise ValueError("All values in the spectrum are NaN, skipping this spaxel.")

    try:
        p0 = [flux_cut.max(), window_centre, std, np.median(flux_cut)]
        popt, _ = curve_fit(gaussian, wave_cut, flux_cut, p0=p0, bounds=bounds)
        lambda_obs = popt[1]
        velocity = c.to(u.km / u.s).value * (lambda_obs - window_centre) / window_centre
    except:
        raise RuntimeError("Curve fitting failed for the spectrum at coordinates ({}, {}).".format(y, x))
    return velocity, flux_cut, wave_cut, popt

def has_peak(spectrum, threshold=0.1):
    return np.any(spectrum > threshold)

def gaussian(w, amp, mu, sigma, offset):
    return offset + amp * np.exp(-(w - mu)**2 / (2 * sigma**2))