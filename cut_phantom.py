import gecatsim as gcs
from PIL import Image, ImageDraw
import json
import numpy as np

desc = json.load(open("cfg/adult_male_50percentile_chest.json"))

N_fm = len(desc["volumefractionmap_filename"])

h_max, w_max = 1050, 1700


NSLICES = 4
FOV = 1280, 920
FOV_R = FOV[0] // 2, FOV[1] // 2
FOV_S = 0, 0

layout_img = Image.new("RGBA", (FOV[0], FOV[1]), color=(0, 0, 0, 0))


for idx in range(N_fm):
    w, h = desc["cols"][idx], desc["rows"][idx]
    x_offset, y_offset, z_offset = (
        desc["x_offset"][idx],
        desc["y_offset"][idx],
        desc["z_offset"][idx],
    )

    slices = desc["slices"][idx]
    z_offset = int(desc["z_offset"][idx])
    fname = "cfg/" + desc["volumefractionmap_filename"][idx]

    print(f"Reading {fname}")
    raw = gcs.rawread(fname, (slices, h, w), "int8")

    fl_center = desc["cols"][1] // 2, desc["rows"][1] // 2
    center = desc["cols"][idx] // 2, desc["rows"][idx] // 2

    center_offset = (((w + 1) / 2 - x_offset), ((h + 1) / 2 - y_offset))
    point_offset = w / 2 - center_offset[0], h / 2 + center_offset[1]
    mountpoint = int(w_max / 2 - point_offset[0]), int(h_max / 2 - point_offset[1])
    isocenter = w_max // 2, h_max // 2

    material = np.zeros((NSLICES * 2, h_max, w_max), dtype=np.int8)

    material[
        :, mountpoint[1] : mountpoint[1] + h, mountpoint[0] : mountpoint[0] + w
    ] = raw[slices // 2 - NSLICES : slices // 2 + NSLICES, :, :]

    del raw

    material = material[
        :,
        isocenter[1] - FOV_R[1] + FOV_S[1] : isocenter[1] + FOV_R[1] + FOV_S[1],
        isocenter[0] - FOV_R[0] + FOV_S[0] : isocenter[0] + FOV_R[0] + FOV_S[0],
    ]

    layer = np.zeros((FOV[1], FOV[0], 4), dtype=np.uint8)
    layer[:, :, 0] = 255 * ((idx + 1) % 2)
    layer[:, :, 1] = 255 * ((idx + 2) % 2)
    layer[:, :, 2] = 127 * ((idx + 3) % 3)
    layer[:, :, 3] = np.astype(material[0, :, :], np.uint8) * 255

    layout_img.alpha_composite(Image.fromarray(layer))

    out_fname = fname + ".raw"

    gcs.rawwrite(out_fname, np.ascontiguousarray(material))

    print(f"{out_fname} written")

layout_img.save(".temp/layout_img.png")
