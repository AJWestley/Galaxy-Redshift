from scipy.optimize import curve_fit
import numpy as np
from astropy.constants import c
import astropy.units as u

def cube_fit_element(flux, wave, lambda_rest=6562.8, window_centre=6562.8, ws=40, threshold=0.1, std=0.5):
    _, xrange, yrange = flux.shape
    vel_map = np.full((yrange, xrange), np.nan)
    z_map = np.full((yrange, xrange), np.nan)
    flux_map = {}
    wave_map = {}
    popt_map = {}

    for y in range(yrange):
        for x in range(xrange):
            try:
                velocity, flux_cut, wave_cut, z, popt = fit_element(x, y, flux, wave, lambda_rest=lambda_rest, window_centre=window_centre, ws=ws, threshold=threshold, std=std)
                vel_map[y, x] = velocity
                z_map[y, x] = z
                flux_map[(y, x)] = flux_cut
                wave_map[(y, x)] = wave_cut
                popt_map[(y, x)] = popt
            except:
                continue

    return vel_map, flux_map, wave_map, z_map, popt_map

def fit_element(x, y, flux, wave, ws=40, threshold=0.1, window_centre=6562.8, lambda_rest=6562.8, std=0.5):
    
    line_mask = (wave > window_centre - ws) & (wave < window_centre + ws)
    wave_cut = wave[line_mask]
    flux_cut = flux[line_mask, y, x]

    peak_flux = wave_cut[flux_cut.argmax()]

    bounds = (
        [0, peak_flux-10, 0.1, -np.inf],
        [np.inf, peak_flux+10, 3, np.inf]
    )

    if not has_peak(flux_cut, threshold=threshold): 
        raise ValueError("No significant peak found in the spectrum.")
    
    if np.isnan(flux_cut).all(): 
        raise ValueError("All values in the spectrum are NaN, skipping this spaxel.")

    try:
        p0 = [flux_cut.max(), window_centre, std, np.median(flux_cut)]
        popt, _ = curve_fit(gaussian, wave_cut, flux_cut, p0=p0, bounds=bounds)
        lambda_obs = popt[1]
        z = lambda_obs / lambda_rest - 1
        # velocity = c.to(u.km / u.s).value * (lambda_obs - lambda_rest) / lambda_rest
        velocity = c.to(u.km / u.s).value * ((1 + z)**2 - 1) / ((1 + z)**2 + 1) 
    except:
        raise RuntimeError("Curve fitting failed for the spectrum at coordinates ({}, {}).".format(y, x))
    
    return velocity, flux_cut, wave_cut, z, popt

def has_peak(spectrum, threshold=0.1):
    zeroed_spectrum = spectrum - np.nanmin(spectrum)
    return np.any(zeroed_spectrum > threshold)

def gaussian(w, amp, mu, sigma, offset):
    return offset + amp * np.exp(-(w - mu)**2 / (2 * sigma**2))