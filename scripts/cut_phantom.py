import gecatsim as gcs
from PIL import Image
import json
from os import path
import numpy as np

SRC_PATH = ".temp/uncut"
S_H, S_W = 1050, 1700


OUT_PATH = "./cfg/layers"
RSLICES = 4
FOV = 1280, 920
FOV_R = FOV[0] // 2, FOV[1] // 2
FOV_S = 0, 50

desc = json.load(open(path.join(SRC_PATH, "adult_male_50percentile_chest.json")))

layout_img = Image.new("RGBA", (FOV[0], FOV[1]), color=(0, 0, 0, 0))

for idx in range(len(desc["volumefractionmap_filename"])):
    w, h = desc["cols"][idx], desc["rows"][idx]
    slices = desc["slices"][idx]
    x_offset, y_offset, z_offset = (
        desc["x_offset"][idx],
        desc["y_offset"][idx],
        desc["z_offset"][idx],
    )
    fpath = path.join(SRC_PATH, desc["volumefractionmap_filename"][idx])

    print(f"Reading {fpath}")
    raw = gcs.rawread(fpath, (slices, h, w), "int8")

    fl_center = desc["cols"][1] // 2, desc["rows"][1] // 2
    center = desc["cols"][idx] // 2, desc["rows"][idx] // 2

    mountpoint = int(S_W / 2 - x_offset), int(S_H / 2 - h + y_offset)
    isocenter = S_W // 2, S_H // 2

    material = np.zeros((RSLICES * 2, S_H, S_W), dtype=np.int8)

    material[
        :, mountpoint[1] : mountpoint[1] + h, mountpoint[0] : mountpoint[0] + w
    ] = raw[slices // 2 - RSLICES : slices // 2 + RSLICES, :, :]

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

    out_fpath = (fpath + ".raw").replace(SRC_PATH, OUT_PATH)
    print(f"Writing {out_fpath}")
    gcs.rawwrite(out_fpath, np.ascontiguousarray(material))

layout_img.save(path.join(SRC_PATH, "ph_layout.png"))
