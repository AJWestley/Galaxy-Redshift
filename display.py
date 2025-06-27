import numpy as np
import matplotlib.pyplot as plt
import random

def plot_velocity_map(vel_map, vmin=-300, vmax=300, cb_label="Velocity (m/s)", title=None):
    plt.imshow(vel_map, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label=cb_label)
    plt.title(title)
    plt.show()

def plot_example_spectra(flux, wave, n=1, title='Example Spectra'):

    _, xmax, ymax = flux.shape

    plt.figure(figsize=(10, 5))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')

    for _ in range(5):
        x = random.randint(0, xmax - 1)
        y = random.randint(0, ymax - 1)
        spectrum = flux[:, x, y]
        plt.plot(wave, spectrum)
    plt.title(title)
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
    return np.abs(array - value).argmin()

def normalize(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img -= img.min()
    img /= img.max()
    return img