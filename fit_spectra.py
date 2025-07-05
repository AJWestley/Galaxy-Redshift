from scipy.optimize import curve_fit
import numpy as np
from astropy.constants import c
import astropy.units as u

def cube_fit_element(flux, wave, lambda_rest=6562.8, window_centre=6562.8, ws=40, threshold=0.1, std=0.5):
    '''
    Takes in a spatially resolved spectrum, and attempts to fit a Gaussian to the emission line at the specified rest wavelength.

    Parameters
    ----------
    flux : np.ndarray
        3D array of shape (n_wave, n_y, n_x) containing the flux values of the spectrum.
    
    wave : np.ndarray
        1D array of shape (n_wave,) containing the wavelength values corresponding to the flux
        values.

    lambda_rest : float, optional
        The rest wavelength of the emission line to fit. Default is 6562.8 (H-alpha).

    window_centre : float, optional
        The central wavelength around which to fit the Gaussian. Default is 6562.8 (H-alpha).

    ws : float, optional
        The width of the window around the central wavelength to consider for fitting. Default is 40
        (in Angstroms).

    threshold : float, optional
        The threshold for detecting a significant peak in the spectrum. Default is 0.1.

    std : float, optional
        The initial guess for the standard deviation of the Gaussian fit. Default is 0.5.

    Returns
    -------
    vel_map : np.ndarray
        2D array of shape (n_y, n_x) containing the velocity values for each spaxel.

    flux_map : dict
        Dictionary mapping (y, x) coordinates to the flux values of the fitted Gaussian for each
        spaxel.

    wave_map : dict
        Dictionary mapping (y, x) coordinates to the wavelength values used for the fit for each
        spaxel.

    z_map : np.ndarray
        2D array of shape (n_y, n_x) containing the redshift values for each spaxel.

    popt_map : dict
        Dictionary mapping (y, x) coordinates to the optimal parameters of the Gaussian fit for each
        spaxel.
    '''
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
    '''
    Fits a Gaussian to the emission line at the specified rest wavelength for a given spaxel.

    Parameters
    ----------
    x : int
        The x-coordinate of the spaxel to fit.

    y : int
        The y-coordinate of the spaxel to fit.

    flux : np.ndarray
        3D array of shape (n_wave, n_y, n_x) containing the flux values of the spectrum.

    wave : np.ndarray
        1D array of shape (n_wave,) containing the wavelength values corresponding to the flux
        values.

    ws : float, optional
        The width of the window around the central wavelength to consider for fitting. Default is 40.

    threshold : float, optional
        The threshold for detecting a significant peak in the spectrum. Default is 0.1.

    window_centre : float, optional
        The central wavelength around which to fit the Gaussian. Default is 6562.8 (H-alpha).

    lambda_rest : float, optional
        The rest wavelength of the emission line to fit. Default is 6562.8 (H-alpha).

    std : float, optional
        The initial guess for the standard deviation of the Gaussian fit. Default is 0.5.

    Returns
    -------
    velocity : float
        The velocity of the emission line in km/s.

    flux_cut : np.ndarray
        1D array containing the flux values of the fitted Gaussian for the spaxel.

    wave_cut : np.ndarray
        1D array containing the wavelength values used for the fit for the spaxel.

    z : float
        The redshift of the emission line.

    popt : np.ndarray
        1D array containing the optimal parameters of the Gaussian fit for the spaxel.
    '''
    
    line_mask = (wave > window_centre - ws) & (wave < window_centre + ws)
    wave_cut = wave[line_mask]
    flux_cut = flux[line_mask, y, x]

    peak_flux = wave_cut[flux_cut.argmax()]

    bounds = (
        [0, peak_flux-ws, 0.1, -np.inf],
        [np.inf, peak_flux+ws, 3, np.inf]
    )

    if not has_peak(flux_cut, threshold=threshold): 
        raise ValueError("No significant peak found in the spectrum.")
    
    if np.isnan(flux_cut).all(): 
        raise ValueError("All values in the spectrum are NaN, skipping this spaxel.")

    try:
        p0 = [flux_cut.max(), peak_flux, std, np.median(flux_cut)]
        popt, _ = curve_fit(gaussian, wave_cut, flux_cut, p0=p0, bounds=bounds)
        lambda_obs = popt[1]
        z = lambda_obs / lambda_rest - 1
        # velocity = c.to(u.km / u.s).value * (lambda_obs - lambda_rest) / lambda_rest
        velocity = c.to(u.km / u.s).value * ((1 + z)**2 - 1) / ((1 + z)**2 + 1) 
    except:
        raise RuntimeError("Curve fitting failed for the spectrum at coordinates ({}, {}).".format(y, x))
    
    return velocity, flux_cut, wave_cut, z, popt

def has_peak(spectrum, threshold=0.1):
    '''
    Checks if the spectrum has a significant peak above a given threshold.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array containing the flux values of the spectrum.

    threshold : float, optional
        The threshold for detecting a significant peak in the spectrum. Default is 0.1.

    Returns
    -------
    bool
        True if the spectrum has a significant peak above the threshold, False otherwise.
    '''
    zeroed_spectrum = spectrum - max(np.nanmin(spectrum), 0)
    return np.any(zeroed_spectrum > threshold)

def gaussian(w, amp, mu, sigma, offset):
    '''Fits a Gaussian function to the data.'''
    return offset + amp * np.exp(-(w - mu)**2 / (2 * sigma**2))