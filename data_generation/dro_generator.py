import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import itertools
import json

from data_generation.utils import LesionBBox

# Set of Default Parameters
default_parameters = {
    # Size Features
    "mean_radius": [100, 100, 1],
    # Shape Features
    "x_deformation": [1, 1, 1],
    "y_deformation": [1, 1, 1],
    "z_deformation": [1, 1, 1],
    "surface_frequency": [0, 0, 1],
    "surface_amplitude": [0, 0, 1],
    # Intensity Features
    "mean_intensity": [100, 100, 1],
    # Texture Features
    "texture_wavelength": [0, 0, 1],
    "texture_amplitude": [0, 0, 1],
    # Margin Features
    "gaussian_standard_deviation": [0, 0, 1],
}

# Keys in the Order for the DRO Name
ordered_keys = [
    "mean_radius",
    "x_deformation",
    "y_deformation",
    "z_deformation",
    "surface_frequency",
    "surface_amplitude",
    "mean_intensity",
    "texture_wavelength",
    "texture_amplitude",
    "gaussian_standard_deviation",
]


# expand_range
# Takes:    dictionary of parameters
# Does:     expands the min, max, number of values into array of values at equal intervals
# Returns:  dictionary of parameters with full arrays of values
def expand_range(dic):
    expanded = {}
    for key in dic.keys():
        kmax = dic[key][0]
        kmin = dic[key][1]
        knum = dic[key][2]
        expanded[key] = frange(kmin, kmax, knum)
    return expanded


# generate_params
# Takes:    dictionary of parameters with full arrays of values
# Does:     find all combinations of parameters of all ranges of values
# Returns:  array of all combinations of parameters
def generate_params(dic):
    params = []
    for key in ordered_keys:
        params.append(dic[key])
    params = list(itertools.product(*params))
    params = [list(p) for p in params]
    return params


def get_single_dro(arguments):
    arguments = [float(arg) for arg in arguments]
    global r, xx, yy, zz, shape_freq, shape_amp, avg, text_wav, text_amp, decay
    r, xx, yy, zz, shape_freq, shape_amp, avg, text_wav, text_amp, decay = arguments
    mask, output_array = generate_dro()
    return output_array, mask


def get_all_dros(params) -> list[tuple[np.ndarray, np.ndarray]]:
    """Creates DROs without writing anything to disk

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: [(DRO, mask)]
    """
    return [get_single_dro(param) for param in params]


# generate_dro
# Takes:    nothing
# Does:     generate dro from its mathematical definition
# Return:   image array embedding the object and mask for the object
def generate_dro():
    n = 300
    s = 512
    # Make 3D Grid
    x = np.linspace(-s / 2, s / 2, s)
    y = np.linspace(-s / 2, s / 2, s)
    z = np.linspace(-n / 2, n / 2, n)
    xt, yt, zt = np.meshgrid(x, y, z, sparse=True)  # xt stands for "x-true"
    if xx != 1 or yy != 1 or zz != 1:
        xs, ys, zs = np.meshgrid(
            1 / float(xx) * x, 1 / float(yy) * y, 1 / float(zz) * z, sparse=True
        )  # xs stands for "x stretch"
    else:
        xs, ys, zs = xt, yt, zt
    # Calculate distance to origin of each point then compare to the shape of the object
    origin = np.sqrt(xs * xs + ys * ys + zs * zs)
    rp = r
    if shape_amp != 0.0 and shape_freq != 0.0:
        rp = r * (
            1
            + shape_amp
            * np.sin(shape_freq * np.arccos(zs / origin))
            * np.cos(shape_freq * np.arctan2(ys, xs))
        )
    mask = rp >= origin
    # Apply Texture
    texture = np.full_like(mask, 1024, dtype=float)
    if text_amp != 0.0 and text_wav != 0.0:
        variation = avg + text_amp * np.cos((1 / text_wav) * 2 * np.pi * xt) * np.cos(
            (1 / text_wav) * 2 * np.pi * yt
        ) * np.cos((1 / text_wav) * 2 * np.pi * zt)
        texture += variation
    else:
        texture += avg
    # Add blurred edge
    if decay != 0:
        big = binary_dilation(mask, iterations=10)
        texture[~big] = 0
        inside = np.copy(texture)
        inside[~mask] = 0
        texture = gaussian_filter(texture, sigma=decay)
        output_array = texture
        texture[mask] = 0
        output_array = inside + texture
    else:
        texture[~mask] = 0
        output_array = texture
    return mask, output_array


# Create a numpy range
def frange(start, stop, step):
    return np.linspace(start, stop, num=step).tolist()


def read_json_cfg(path: str) -> list[dict]:
    params = []
    orbit = 150
    with open(path, "r", encoding="utf-8") as cfg_file:
        cfg_content = json.load(cfg_file)
        orbit = cfg_content["orbit"]
        for object in cfg_content["objects"]:
            obj_param = dict(default_parameters.items())
            for key, val in object.items():
                obj_param[key] = [val, val, 1]
            params.append(obj_param)
    return params, orbit


def generate_tex(scale: int, cutoff: float, shape):
    tex_shape = (
        shape[0] // scale,
        shape[1] // scale,
        shape[2] // scale,
    )

    bg_tex = np.random.uniform(0, 1, tex_shape) > cutoff

    if scale > 1:
        bg_tex = np.kron(bg_tex, np.ones((scale, scale, scale)))

    return bg_tex


def generate_phantom(
    cfg_path: str,
    roi_radius: int,
    shape: tuple[int],
) -> tuple[np.ndarray, np.ndarray, list[LesionBBox]]:
    """Generates a phantom.

    Args:
        cfg_path (str): path to JSON with custom config.

    Returns:
        tuple[np.ndarray, list[LesionBBox]]: phantom mask and lesions on it.
    """

    phantom = np.zeros(shape, dtype=np.bool)
    ph_mid = shape[0] // 2, shape[1] // 2, shape[2] // 2
    traj_deform = shape[1] / shape[0]

    # objects_cfgs, texture_cfg = read_json_cfg(cfg_path)
    objects_cfgs, orbit = read_json_cfg(cfg_path)
    # tex_cutoff = texture_cfg["noise_cutoff"]

    angle_s = 2 * np.pi / (len(objects_cfgs) - 1)
    safe_r = int(orbit * np.cos(np.pi / 2 - angle_s / 2))

    if safe_r <= roi_radius:
        raise UserWarning(f"Safezone too small for ROI ({safe_r} < {roi_radius})")

    bboxes = []

    for i, ocfg in enumerate(objects_cfgs):
        obj_mean_r = ocfg["mean_radius"][0]
        obj_amplitude = ocfg["surface_amplitude"][0]
        obj_x_deform = ocfg["x_deformation"][0]
        obj_y_deform = ocfg["y_deformation"][0]
        obj_z_deform = ocfg["z_deformation"][0]

        assert (
            max(obj_x_deform, obj_y_deform, obj_z_deform) <= 1.0
        ), "Enlarging DRO deformation unsupported"

        obj_r = obj_mean_r + int(obj_mean_r * obj_amplitude)

        if safe_r <= obj_r:
            raise UserWarning(
                f"Object might not fit into safezone ({safe_r} <= {obj_r})"
            )

        if roi_radius <= obj_r:
            raise UserWarning(
                f"Object might not fit into ROI ({roi_radius} <= {obj_r})"
            )

        _, mask = get_single_dro(generate_params(expand_range(ocfg))[0])

        m_zcut = mask.shape[2] // 2
        m_zcut_r = shape[2] // 2

        mask = mask[
            256 - safe_r : 256 + safe_r,
            256 - safe_r : 256 + safe_r,
            m_zcut - m_zcut_r : m_zcut + m_zcut_r,
        ]

        xc = int(orbit * traj_deform * np.cos(-angle_s * (i - 1))) + ph_mid[1]
        yc = int(orbit * np.sin(-angle_s * (i - 1))) + ph_mid[0]

        if i == 0:
            xc, yc = ph_mid[1], ph_mid[0]

        xi, yi = xc - safe_r, yc - safe_r
        xo, yo = xc + safe_r, yc + safe_r

        phantom[yi:yo, xi:xo, :] = np.logical_or(mask, phantom[yi:yo, xi:xo, :])

        # TODO: bring back real safe r
        # outer_circle_r = min(ph_mid[0], ph_mid[1]) - np.sqrt((xc - ph_mid[1]) ** 2 + (yc - ph_mid[0]) ** 2)
        # real_safe_r = min(safe_r, int(outer_circle_r / np.sqrt(2)))
        real_safe_r = safe_r
        r_z = int(obj_r * obj_z_deform)

        bboxes.append(
            LesionBBox(
                center=(xc, yc, ph_mid[2]),
                r=(
                    int(obj_r * obj_x_deform),
                    int(obj_r * obj_y_deform),
                    r_z,
                ),
                safe_r=(real_safe_r, real_safe_r, r_z),
                roi_r=(roi_radius, roi_radius, r_z),
            )
        )

    # bg_tex = generate_tex(4, tex_cutoff, phantom.shape)
    # bg_tex = np.logical_or(bg_tex, generate_tex(2, tex_cutoff, phantom.shape))
    # bg_tex = np.logical_or(bg_tex, generate_tex(1, tex_cutoff, phantom.shape))

    # TODO: Deprecate
    bg_tex = np.zeros(shape, dtype=np.bool)

    return phantom, bboxes, bg_tex
