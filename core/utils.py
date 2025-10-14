import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import constants


def calculate_wavelength(photon_energy):
    """Calculate the wavelength from photon energy."""
    hc = constants.value(
        'inverse meter-electron volt relationship')  # h * c in eV * m
    return hc / photon_energy


def center_crop(img, target_size):
    """
    Center-crop a 2D image array to target_size x target_size.
    If the image is smaller than target_size in a dimension, that dimension is left as-is.
    Returns the cropped image.
    """
    height, width = img.shape
    if height >= target_size:
        start_h = (height - target_size) // 2
        end_h = start_h + target_size
    else:
        start_h = 0
        end_h = height

    if width >= target_size:
        start_w = (width - target_size) // 2
        end_w = start_w + target_size
    else:
        start_w = 0
        end_w = width

    return img[start_h:end_h, start_w:end_w]


def image_correction(image, flat=None, dark=None, epsilon=1e-8, normalize=True):
    f32 = np.float32
    if flat is None and dark is None:
        out = image.astype(f32, copy=True)
    elif flat is None:
        out = image.astype(f32, copy=True); out -= dark.astype(f32, copy=False)
    else:
        num = image.astype(f32, copy=False) - (dark.astype(f32, copy=False) if dark is not None else 0)
        den = (flat.astype(f32, copy=False) - (dark.astype(f32, copy=False) if dark is not None else 0))
        mask = den > epsilon;
        out = np.zeros_like(num, dtype=f32);
        np.divide(num, den + epsilon, out=out, where=mask)
    if normalize:
        cnt = np.count_nonzero(out) if flat is not None else out.size
        if cnt > 0: out /= (out.sum(dtype=np.float64) / cnt)
    return out


def load_images(image_path, dark_image_path=None, flat_image_path=None):
    with Image.open(image_path) as im:
        image = np.array(im, dtype=np.float32)
    dark = None if not dark_image_path else np.array(Image.open(dark_image_path), dtype=np.float32)
    flat = None if not flat_image_path else np.array(Image.open(flat_image_path), dtype=np.float32)
    return image, dark, flat


def plot_distance_relationship(params):
    """
    Plot a schematic diagram of the distance relationship, showing the positional relationship between the source, sample, and detector

    Parameters:
    - params: A dictionary containing calibrated distance parameters
    - old_source_dist: Source distance before calibration
    - total_dist: Total distance
    - base_path: Save path
    """
    # import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Top plot: distance relationship before calibration
    ax.set_xlim(-1, params["total_dist"] + 1)
    ax.set_ylim(-0.2, 1.5)

    # Draw the optical path
    ax.plot([-1, params["total_dist"] + 1], [0.5, 0.5], 'k--', linewidth=2, alpha=0.3)

    # Mark position points
    source_pos = 0
    sample_pos_old = params["old_source_dist"]
    det_pos = params["old_source_dist"] + params["det2sample"]
    delta = params["old_source_dist"] - params["source_dist"]
    # Draw each component

    ax.plot(source_pos, 0.5, 'o', color='gray', markersize=10, label='Zero point')
    ax.plot(delta, 0.5, 'ro', markersize=10, label='Focus point')
    ax.plot([sample_pos_old, sample_pos_old], [0.4, 0.6], 'b-', linewidth=3, label='Sample')
    ax.plot([det_pos, det_pos], [0.4, 0.6], 'g-', linewidth=3, label='Detector')

    # Add distance annotations
    ax.annotate('', xy=(sample_pos_old, 0.4), xytext=(source_pos, 0.4),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax.text((source_pos + sample_pos_old) / 2, 0.35,
            f'old_source_dist = {sample_pos_old:.3f} m',
            ha='center', va='top', fontsize=10, color='gray')

    ax.annotate('', xy=(det_pos, 0.4), xytext=(sample_pos_old, 0.4),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text((sample_pos_old + det_pos) / 2, 0.35,
            f'det2sample = {params["det2sample"]:.3f} m',
            ha='center', va='bottom', fontsize=10, color='blue')

    ax.annotate('', xy=(delta, 0.65), xytext=(source_pos, 0.65),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text((delta + source_pos) / 2, 0.60,
            f'focus_adjust = {delta:.3f} m',
            ha='center', va='bottom', fontsize=10, color='green')

    ax.annotate('', xy=(det_pos, 0.65), xytext=(delta, 0.65),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((delta + det_pos) / 2, 0.7,
            f'total_dist = {params["total_dist"]:.3f} m',
            ha='center', va='bottom', fontsize=10, color='red')

    ax.set_title('Distance Relationship', fontsize=14)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()
