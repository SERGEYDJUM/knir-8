import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path as catsim_paths

from PIL import Image, ImageDraw
from numpy.typing import NDArray
from os import path, makedirs
from argparse import ArgumentParser, Namespace
from datetime import datetime
from hashlib import md5
import numpy as np
import csv
import json

from .utils import Phantom, save_raw
from .dro_generator import generate_phantom

WORKDIR = path.abspath("./.temp")
CFGDIR = path.abspath("./cfg")
DATASET_DIR = path.abspath("./dataset")
EXPERIMENT_PREFIX = path.join(WORKDIR, "MAIN")

CSV_HEADER = (
    "tomogram_index",
    "slice_index",
    "bbox_index",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_radius_x",
    "bbox_radius_y",
    "bbox_safe_r_x",
    "bbox_safe_r_y",
    "signal_present",
    "tube_current",
    "recon_kernel",
    "phantom_cfg_md5",
    "xcist_cfg_md5",
    "raw_source",
)


def process_signals(mask, bboxes, bg_tex_mask, save_mask: bool = False) -> Phantom:
    z_slices = mask.shape[0]
    size = mask.shape[2]

    # x, y = np.mgrid[:size, :size]
    # allowed_circle = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 2) ** 2

    empty_mask = np.ones_like(bg_tex_mask) - bg_tex_mask
    save_raw(empty_mask, path.join(CFGDIR, "dro_phantom_bg_empty.raw"))
    save_raw(bg_tex_mask, path.join(CFGDIR, "dro_phantom_t_empty.raw"))

    if save_mask:
        img = Image.fromarray((mask[z_slices // 2, :, :].astype(np.uint8) * 255))
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(bbox.bbox(), outline=192)
            draw.rectangle(bbox.roi_bbox(), outline=96)
            draw.rectangle(bbox.safe_bbox(), outline=32)

        img.save(path.join(WORKDIR, f"mask_{z_slices // 2}.png"))

    bg_tex_mask = np.clip(bg_tex_mask - mask, 0, 1)
    inv_mask = 1 - mask - bg_tex_mask
    inv_mask = np.clip(inv_mask, 0, 1)

    # mask *= allowed_circle
    # inv_mask *= allowed_circle
    # bg_tex_mask *= allowed_circle

    if save_mask:
        img = Image.fromarray((bg_tex_mask[z_slices // 2, :, :].astype(np.uint8) * 255))
        img.save(path.join(WORKDIR, f"tex_mask_{z_slices // 2}.png"))

    save_raw(mask, path.join(CFGDIR, "dro_phantom_mask.raw"))
    save_raw(inv_mask, path.join(CFGDIR, "dro_phantom_water_mask.raw"))
    save_raw(bg_tex_mask, path.join(CFGDIR, "dro_phantom_tex_mask.raw"))

    return Phantom(bboxes)


def optimize_layer(
    layer: NDArray, x_offset: int, y_offset: int
) -> tuple[NDArray, int, int]:
    ss_layer = layer.sum(axis=0)
    rowsum = ss_layer.sum(axis=0) > 0
    colsum = ss_layer.sum(axis=1) > 0

    r_start, r_end = 0, ss_layer.shape[0]
    c_start, c_end = 0, ss_layer.shape[1]

    for i in range(0, len(colsum)):
        if colsum[i]:
            r_start = max(i - 1, 0)
            break

    for i in reversed(range(0, len(colsum))):
        if colsum[i]:
            r_end = i + 1
            break

    for j in range(0, len(rowsum)):
        if rowsum[j]:
            c_start = max(j - 1, 0)
            break

    for j in reversed(range(0, len(rowsum))):
        if rowsum[j]:
            c_end = j + 1
            break

    return (
        np.ascontiguousarray(layer[:, r_start:r_end, c_start:c_end]),
        x_offset - c_start,
        r_end - y_offset,
    )


def patched_phantom(save_mask: bool = False):
    ph_g_path = path.join(CFGDIR, "Phantom_Generation.json")
    b_p_path = path.join(CFGDIR, "Base_Phantom_Descriptor.json")

    phantom_cfg = json.load(open(b_p_path))
    json.dump(phantom_cfg, open(path.join(CFGDIR, "Phantom_NS_Descriptor.json"), "w"))
    material = json.load(open(ph_g_path))["material"]

    m_slices = int(max(phantom_cfg["slices"]))
    m_rows = int(max(phantom_cfg["rows"]))
    m_cols = int(max(phantom_cfg["cols"]))

    mask, bboxes, bg_tex_mask = generate_phantom(
        ph_g_path, roi_radius=64, shape=(m_rows, m_cols, m_slices)
    )

    inv_mask = np.transpose(np.logical_not(mask).astype(np.int8), (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1)).astype(np.int8)
    bg_tex_mask = np.transpose(bg_tex_mask, (2, 0, 1)).astype(np.int8)
    bboxes = process_signals(mask, bboxes, bg_tex_mask, save_mask=save_mask)

    for i, layer_name in enumerate(phantom_cfg["volumefractionmap_filename"]):
        x_offset, y_offset = phantom_cfg["x_offset"][i], phantom_cfg["y_offset"][i]
        slices, rows, cols = (
            phantom_cfg["slices"][i],
            phantom_cfg["rows"][i],
            phantom_cfg["cols"][i],
        )

        layer = xc.rawread(path.join(CFGDIR, layer_name), (slices, rows, cols), "int8")
        midpoint = layer.shape[1] // 2, layer.shape[2] // 2
        mask_r = mask.shape[1] // 2, mask.shape[2] // 2

        layer[
            :,
            midpoint[0] - mask_r[0] : midpoint[0] + mask_r[0],
            midpoint[1] - mask_r[1] : midpoint[1] + mask_r[1],
        ] *= inv_mask

        layer, x_offset, y_offset = optimize_layer(layer, x_offset, y_offset)

        phantom_cfg["x_offset"][i] = x_offset
        phantom_cfg["y_offset"][i] = y_offset
        phantom_cfg["rows"][i] = layer.shape[1]
        phantom_cfg["cols"][i] = layer.shape[2]

        xc.rawwrite(path.join(CFGDIR, layer_name + ".patched"), layer)

    phantom_cfg["n_materials"] += 1
    phantom_cfg["mat_name"].append(material)

    vfms = list(
        map(lambda x: x + ".patched", phantom_cfg["volumefractionmap_filename"])
    )

    phantom_cfg["volumefractionmap_filename"] = vfms + ["dro_phantom_mask.raw"]
    phantom_cfg["volumefractionmap_datatype"].append("int8")
    phantom_cfg["cols"].append(mask.shape[2])
    phantom_cfg["rows"].append(mask.shape[1])
    phantom_cfg["slices"].append(mask.shape[0])
    phantom_cfg["x_size"].append(phantom_cfg["x_size"][-1])
    phantom_cfg["y_size"].append(phantom_cfg["y_size"][-1])
    phantom_cfg["z_size"].append(phantom_cfg["z_size"][-1])
    phantom_cfg["x_offset"].append(mask.shape[2] // 2)
    phantom_cfg["y_offset"].append(mask.shape[1] // 2)
    phantom_cfg["z_offset"].append(mask.shape[0] // 2)

    json.dump(phantom_cfg, open(path.join(CFGDIR, "Phantom_Descriptor.json"), "w"))

    # TODO: Fix bboxes after patching

    return bboxes


def reconstruct(catsim: xc.CatSim) -> tuple[NDArray, float]:
    catsim.do_Recon = 1
    recon.recon(catsim)

    imsize, slice_cnt = catsim.recon.imageSize, catsim.recon.sliceCount
    bbox_scale = catsim.phantom.scale * (imsize**2 / (512 * catsim.recon.fov))

    tomogram = xc.rawread(
        EXPERIMENT_PREFIX + f"_{imsize}x{imsize}x{slice_cnt}.raw",
        [slice_cnt, imsize, imsize],
        "float",
    )

    return tomogram, bbox_scale


def demo_reconstructed(
    tomogram: NDArray, bboxes: Phantom, name_prefix: str = "recon"
) -> None:
    demo_dir = path.join(WORKDIR, "out")
    makedirs(demo_dir, exist_ok=True)

    demo_tomo = np.clip(tomogram, -1000, 1000)
    demo_tomo += 1000
    demo_tomo *= 255 / 2000
    demo_tomo = np.array(demo_tomo, dtype=np.uint8)

    for i in range(demo_tomo.shape[0]):
        image = Image.fromarray(demo_tomo[i, :, :]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for bbox in bboxes.signals:
            # draw.rectangle(bbox.bbox(), outline=(0, 255, 0))
            draw.rectangle(bbox.roi_bbox(), outline=(0, 0, 0))
            draw.rectangle(bbox.safe_bbox(), outline=(0, 0, 255))
        image.save(path.join(demo_dir, f"{name_prefix}_slice_{i}.png"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--demo", action="store_true")
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("-s", "--skipgen", action="store_true")
    parser.add_argument("-e", "--empty", action="store_true")
    parser.add_argument("--reconstruct-only", action="store_true")
    parser.add_argument("-r", "--repeat", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    catsim_paths.add_search_path(path.abspath(CFGDIR))

    bbox_cache = path.join(WORKDIR, "bboxes.json")
    catsim_cfg = path.join(CFGDIR, "CatSim.cfg")
    phantomgen_cfg = path.join(CFGDIR, "Phantom_Generation.json")
    base_phantom_cfg = path.join(CFGDIR, "Base_Phantom_Descriptor.json")

    dataset_csv = path.join(DATASET_DIR, "dataset.csv")
    dataset_cfgs = path.join(DATASET_DIR, "cfgs")
    dataset_raws = path.join(DATASET_DIR, "raws")

    makedirs(WORKDIR, exist_ok=True)
    makedirs(dataset_cfgs, exist_ok=True)
    makedirs(dataset_raws, exist_ok=True)

    # Phantom generation
    bboxes: Phantom = None
    if not args.skipgen:
        bboxes = patched_phantom(save_mask=args.mask)
        bboxes.dump(bbox_cache)
    else:
        bboxes = Phantom.load(bbox_cache)

    cfg_timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "")
    ph_ds_path = path.join(dataset_cfgs, f"{cfg_timestamp}.json")
    ph_cfg_hash: str = None

    with (
        open(phantomgen_cfg, "rb") as ifile,
        open(ph_ds_path, "wb") as ofile,
    ):
        contents = ifile.read()
        ph_cfg_hash = md5(contents).hexdigest()
        ofile.write(contents)

    cs_ds_path = path.join(dataset_cfgs, f"{cfg_timestamp}.cfg")
    catsim_cfg_hash: str = None

    with (
        open(catsim_cfg, "rb") as ifile,
        open(cs_ds_path, "wb") as ofile,
    ):
        contents = ifile.read()
        catsim_cfg_hash = md5(contents).hexdigest()
        ofile.write(contents)

    bp_cfg = json.load(open(base_phantom_cfg))
    bp_slices = int(max(bp_cfg["slices"]))
    bp_rows = int(max(bp_cfg["rows"]))
    bp_cols = int(max(bp_cfg["cols"]))

    # Simulation
    catsim = xc.CatSim(catsim_cfg)
    if args.empty:
        catsim.phantom.filename = "Phantom_NS_Descriptor.json"

    catsim.protocol.viewCount = catsim.protocol.viewsPerRotation
    catsim.protocol.stopViewId = catsim.protocol.viewCount - 1

    catsim.resultsName = EXPERIMENT_PREFIX

    if not path.exists(dataset_csv):
        with open(dataset_csv, "w", encoding="utf-8") as csvfile:
            csvfile.write(",".join(CSV_HEADER) + "\n")

    with open(dataset_csv, "a", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")

        for repeat_iter in range(args.repeat):
            if not args.reconstruct_only:
                catsim.run_all()

            tomogram, imscale = reconstruct(catsim)

            raw_timestamp = (
                datetime.now().isoformat(timespec="seconds").replace(":", "")
            )
            raw_path = path.join(dataset_raws, f"{raw_timestamp}.raw")

            with open(raw_path, "wb") as raw_file:
                raw_file.write(tomogram)

            bboxes.transform(
                imscale,
                (
                    bp_cols // 2,
                    bp_rows // 2,
                    bp_slices // 2,
                ),
                (
                    tomogram.shape[2] // 2,
                    tomogram.shape[1] // 2,
                    tomogram.shape[0] // 2,
                ),
            )

            for slice_idx in range(tomogram.shape[0]):
                for bb_idx, bbox in enumerate(bboxes.signals):
                    writer.writerow(
                        (
                            repeat_iter,
                            slice_idx,
                            bb_idx,
                            bbox.center[0],
                            bbox.center[1],
                            bbox.r[0],
                            bbox.r[1],
                            bbox.sr[0],
                            bbox.sr[1],
                            not args.empty,
                            int(catsim.protocol.mA),
                            catsim.recon.kernelType,
                            ph_cfg_hash,
                            catsim_cfg_hash,
                            path.relpath(raw_path, dataset_raws),
                        )
                    )

            if args.demo:
                demo_reconstructed(tomogram, bboxes)


if __name__ == "__main__":
    main()
