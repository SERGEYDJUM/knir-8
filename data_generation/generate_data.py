from copy import deepcopy
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
CFG_C_DIR = path.abspath("./cfg")
DATASET_DIR = path.abspath("./dataset")
EXPERIMENT_PREFIX = path.join(WORKDIR, "MAIN")

cfg_dir = path.join(CFG_C_DIR, "cfg_0")


CSV_HEADER = (
    "raw_source",
    "slice_index",
    "bbox_index",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_radius_x",
    "bbox_radius_y",
    "bbox_safe_r_x",
    "bbox_safe_r_y",
    "signal_present",
    "human_score",
    "tube_current",
    "recon_kernel",
    "phantom_cfg_md5",
    "xcist_cfg_md5",
)


def process_signals(mask, bboxes, bg_tex_mask, save_mask: bool = False) -> Phantom:
    z_slices = mask.shape[0]
    size = mask.shape[2]

    # x, y = np.mgrid[:size, :size]
    # allowed_circle = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 2) ** 2

    # empty_mask = np.ones_like(bg_tex_mask) - bg_tex_mask
    # save_raw(empty_mask, path.join(CFGDIR, "dro_phantom_bg_empty.raw"))
    # save_raw(bg_tex_mask, path.join(CFGDIR, "dro_phantom_t_empty.raw"))

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

    save_raw(mask, path.join(CFG_C_DIR, "layers/dro_phantom_mask.raw"))
    # save_raw(inv_mask, path.join(CFGDIR, "dro_phantom_water_mask.raw"))
    # save_raw(bg_tex_mask, path.join(CFGDIR, "dro_phantom_tex_mask.raw"))

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


def opt_write_layer(layer: NDArray, idx: int, cfg: dict, postfix: str) -> dict:
    opt_layer, opt_x_offset, opt_y_offset = optimize_layer(
        layer, cfg["x_offset"][idx], cfg["y_offset"][idx]
    )

    cfg["x_offset"][idx] = opt_x_offset
    cfg["y_offset"][idx] = opt_y_offset
    cfg["rows"][idx] = opt_layer.shape[1]
    cfg["cols"][idx] = opt_layer.shape[2]
    cfg["volumefractionmap_filename"][idx] += postfix

    xc.rawwrite(path.join(CFG_C_DIR, cfg["volumefractionmap_filename"][idx]), opt_layer)
    return cfg


def patched_phantom(save_mask: bool = False):
    global cfg_dir
    ph_g_path = path.join(cfg_dir, "Phantom_Generation.json")
    b_p_path = path.join(CFG_C_DIR, "Base_Phantom_Descriptor.json")

    material = json.load(open(ph_g_path))["material"]
    ph_cfg = json.load(open(b_p_path))
    ns_ph_cfg = deepcopy(ph_cfg)

    m_slices = int(max(ph_cfg["slices"]))
    m_rows = int(max(ph_cfg["rows"]))
    m_cols = int(max(ph_cfg["cols"]))

    mask, bboxes, bg_tex_mask = generate_phantom(
        ph_g_path, roi_radius=64, shape=(m_rows, m_cols, m_slices)
    )

    mask = np.transpose(mask, (2, 0, 1)).astype(np.int8)
    bg_tex_mask = np.transpose(bg_tex_mask, (2, 0, 1)).astype(np.int8)
    bboxes = process_signals(mask, bboxes, bg_tex_mask, save_mask=save_mask)

    inv_mask = 1 - mask

    for i in range(ph_cfg["n_materials"]):
        vfm_fname = ph_cfg["volumefractionmap_filename"][i]
        slices, rows, cols = (
            ph_cfg["slices"][i],
            ph_cfg["rows"][i],
            ph_cfg["cols"][i],
        )

        layer = xc.rawread(
            path.join(CFG_C_DIR, vfm_fname), (slices, rows, cols), "int8"
        )
        midpoint = layer.shape[1] // 2, layer.shape[2] // 2
        mask_r = mask.shape[1] // 2, mask.shape[2] // 2

        ns_ph_cfg = opt_write_layer(layer, i, ns_ph_cfg, ".ns_patched")

        layer[
            :,
            midpoint[0] - mask_r[0] : midpoint[0] + mask_r[0],
            midpoint[1] - mask_r[1] : midpoint[1] + mask_r[1],
        ] *= inv_mask

        ph_cfg = opt_write_layer(layer, i, ph_cfg, ".patched")

    with open(path.join(CFG_C_DIR, "Phantom_NS_Descriptor.json"), "w") as cfg_file:
        json.dump(ns_ph_cfg, cfg_file)

    ph_cfg["n_materials"] += 1
    ph_cfg["mat_name"].append(material)
    ph_cfg["volumefractionmap_filename"].append("layers/dro_phantom_mask.raw")
    ph_cfg["volumefractionmap_datatype"].append("int8")
    ph_cfg["cols"].append(mask.shape[2])
    ph_cfg["rows"].append(mask.shape[1])
    ph_cfg["slices"].append(mask.shape[0])
    ph_cfg["x_size"].append(ph_cfg["x_size"][-1])
    ph_cfg["y_size"].append(ph_cfg["y_size"][-1])
    ph_cfg["z_size"].append(ph_cfg["z_size"][-1])
    ph_cfg["x_offset"].append(mask.shape[2] // 2)
    ph_cfg["y_offset"].append(mask.shape[1] // 2)
    ph_cfg["z_offset"].append(mask.shape[0] // 2)

    with open(path.join(CFG_C_DIR, "Phantom_Descriptor.json"), "w") as cfg_file:
        json.dump(ph_cfg, cfg_file)

    return bboxes


def reconstruct(catsim: xc.CatSim) -> tuple[NDArray, float]:
    catsim.do_Recon = 1
    recon.recon(catsim)

    imsize, slice_cnt = catsim.recon.imageSize, catsim.recon.sliceCount
    bbox_scale = catsim.phantom.scale * imsize / catsim.recon.fov

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
            draw.rectangle(bbox.bbox(), outline=(0, 255, 0))
            draw.rectangle(bbox.roi_bbox(), outline=(0, 0, 255))
            draw.rectangle(bbox.safe_bbox(), outline=(0, 0, 0))
        image.save(path.join(demo_dir, f"{name_prefix}_slice_{i}.png"))

    for i in range(demo_tomo.shape[0]):
        image = Image.fromarray(demo_tomo[i, :, :]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for bbox in bboxes.signals:
            draw.rectangle(bbox.safe_bbox(), outline=(0, 0, 0))
        image.save(path.join(demo_dir, f"{name_prefix}_slice_{i}_sroi.png"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--demo", action="store_true")
    parser.add_argument("-s", "--skipgen", action="store_true")
    parser.add_argument("-e", "--empty", action="store_true")
    parser.add_argument("--recon-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("-r", "--repeat", default=1, type=int)
    parser.add_argument("--cfg-dir", default="cfg_0", type=str)

    return parser.parse_args()


def main():
    global cfg_dir
    args = parse_args()
    cfg_dir = path.join(CFG_C_DIR, args.cfg_dir)
    catsim_paths.add_search_path(path.abspath(CFG_C_DIR))

    bbox_cache = path.join(WORKDIR, "bboxes.json")
    catsim_cfg = path.join(cfg_dir, "CatSim.cfg")
    phantomgen_cfg = path.join(cfg_dir, "Phantom_Generation.json")
    base_phantom_cfg = path.join(CFG_C_DIR, "Base_Phantom_Descriptor.json")

    dataset_csv = path.join(DATASET_DIR, "dataset.csv")
    dataset_cfgs = path.join(DATASET_DIR, "cfgs")
    dataset_raws = path.join(DATASET_DIR, "raws")

    makedirs(WORKDIR, exist_ok=True)
    makedirs(dataset_cfgs, exist_ok=True)
    makedirs(dataset_raws, exist_ok=True)

    # Phantom generation
    bboxes: Phantom = None
    if not args.skipgen:
        bboxes = patched_phantom(save_mask=args.demo)
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
    catsim.resultsName = EXPERIMENT_PREFIX
    catsim.protocol.viewCount = catsim.protocol.viewsPerRotation
    catsim.protocol.stopViewId = catsim.protocol.viewCount - 1
    catsim.phantom.callback = "Phantom_Voxelized"
    catsim.phantom.projectorCallback = "C_Projector_Voxelized"
    catsim.phantom.filename = "Phantom_Descriptor.json"

    if args.empty:
        catsim.phantom.filename = "Phantom_NS_Descriptor.json"

    if not path.exists(dataset_csv):
        with open(dataset_csv, "w", encoding="utf-8") as csvfile:
            csvfile.write(",".join(CSV_HEADER) + "\n")

    with open(dataset_csv, "a", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")

        for repeat_iter in range(args.repeat):
            if not args.recon_only:
                catsim.run_all()

            tomogram, imscale = reconstruct(catsim)

            raw_timestamp = (
                datetime.now().isoformat(timespec="seconds").replace(":", "")
            )
            raw_path = path.join(dataset_raws, f"{raw_timestamp}.raw")

            with open(raw_path, "wb") as raw_file:
                raw_file.write(tomogram)

            n_bboxes = deepcopy(bboxes)
            n_bboxes.transform(
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

            print(f"BBoxes scaled by {imscale:.3f}.")

            for slice_idx in range(tomogram.shape[0]):
                for bb_idx, bbox in enumerate(n_bboxes.signals):
                    if args.no_write:
                        continue

                    writer.writerow(
                        (
                            path.relpath(raw_path, dataset_raws),
                            slice_idx,
                            bb_idx,
                            bbox.center[0],
                            bbox.center[1],
                            bbox.r[0],
                            bbox.r[1],
                            bbox.sr[0],
                            bbox.sr[1],
                            not args.empty,
                            -1,
                            int(catsim.protocol.mA),
                            catsim.recon.kernelType,
                            ph_cfg_hash,
                            catsim_cfg_hash,
                        )
                    )

            if args.demo:
                demo_reconstructed(tomogram, n_bboxes)


if __name__ == "__main__":
    main()
