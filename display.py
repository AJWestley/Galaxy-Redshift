import numpy as np
import matplotlib.pyplot as plt
import random

def plot_velocity_map(vel_map, vmin=-300, vmax=300, cb_label="Velocity (km/s)", title=None):
    '''Plots a 2D velocity map with a colorbar.'''
    plt.imshow(vel_map, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label=cb_label)
    plt.title(title)
    plt.show()

def plot_example_spectra(flux, wave, n=1, title='Example Spectra', co_ords=None, xmin=None, xmax=None):
    '''Plots example spectra from a 3D flux array.'''
    _, xdim, ydim = flux.shape

    plt.figure(figsize=(10, 5))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')

    # Mask wave range
    wave_mask = slice(None)
    if xmin is not None or xmax is not None:
        wave_mask = (wave >= (xmin if xmin is not None else wave.min())) & (wave <= (xmax if xmax is not None else wave.max()))

    if co_ords is not None:
        x, y = co_ords
        spectrum = flux[:, x, y][wave_mask]
        plt.plot(wave[wave_mask], spectrum, label=f'Spectrum at ({x}, {y})')
        plt.legend()
    else:
        for _ in range(n):
            x = random.randint(0, xdim - 1)
            y = random.randint(0, ydim - 1)
            spectrum = flux[:, x, y][wave_mask]
            plt.plot(wave[wave_mask], spectrum)

    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.show()

def plot_rgb_image(
        flux, 
        wave, 
        blue_wl = 3621, 
        green_wl = 4840, 
        red_wl = 6231, 
        r_scale=1.0, 
        g_scale=1.0,
        b_scale=1.0,
        title='"Approximate RGB Image of Galaxy"'
        ):
    '''Plots an RGB image from a 3D flux array based on specified wavelength bands.'''
    

    b_idx = find_nearest_idx(wave, blue_wl)
    g_idx = find_nearest_idx(wave, green_wl)
    r_idx = find_nearest_idx(wave, red_wl)

    R = flux[r_idx, :, :]
    G = flux[g_idx, :, :]
    B = flux[b_idx, :, :]

    R = normalize(R) * r_scale
    G = normalize(G) * g_scale
    B = normalize(B) * b_scale

    rgb_image = np.stack([R, G, B], axis=-1)

    # smooth the image
    plt.imshow(rgb_image, origin='lower', interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

def find_nearest_idx(array, value):
    '''Finds the index of the nearest value in an array to a specified value.'''
    return np.abs(array - value).argmin()

def normalize(img):
    '''Normalizes an image array to the range [0, 1].'''
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img -= img.min()
    img /= img.max()
    return img