from os import system, name, path
from labeling.labeling import rawread, load_dataset, DATASET_RAWS
import numpy as np
from numpy.typing import NDArray


ROI_R: int = 32
PIXEL_MUL = 0.001


class DataStore:
    x: NDArray[np.single] = None
    y: NDArray[np.bool] = None
    hy: NDArray[np.bool] = None
    a: NDArray[np.uint8] = None
    k: NDArray[np.bool] = None
    o: NDArray[np.bool] = None
    s: NDArray[np.uint8] = None

    def __init__(self) -> None:
        df, _ = load_dataset()
        # assert len(df[df["human_score"] == -1]) == 0

        raws = {}

        X = np.zeros((df.shape[0], ROI_R * 2, ROI_R * 2), dtype=np.single)
        y = np.zeros(df.shape[0], dtype=np.bool)
        k = np.zeros(df.shape[0], dtype=np.bool)
        a = np.zeros(df.shape[0], dtype=np.uint8)
        hy = np.zeros(df.shape[0], dtype=np.int8)
        o = np.zeros(df.shape[0], dtype=np.bool)
        l = np.zeros(df.shape[0], dtype=np.bool)
        s = np.zeros(df.shape[0], dtype=np.uint8)

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source

            if raw_name not in raws:
                raw_path = path.join(DATASET_RAWS, raw_name)
                max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
                raws[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

            y[i] = row.signal_present
            hy[i] = row.human_score
            k[i] = row.recon_kernel == "soft"
            a[i] = row.tube_current
            o[i] = row.phantom_cfg_md5 in [
                "3d3fd108138ce807fecba23851bbf61b",
                "dca819d68a465d5977515ae96d3dbb84",
            ]
            s[i] = row.bbox_index
            l[i] = row.xcist_cfg_md5 in [
                "cd7e0f84b35facac6a17c69a697754d2",
                "94de7be616c1e06a7456341593426681",
            ]

            center = row.bbox_center_x, row.bbox_center_y
            X[i] = raws[raw_name][row.slice_index][
                center[1] - ROI_R : center[1] + ROI_R,
                center[0] - ROI_R : center[0] + ROI_R,
            ]

        self.x, self.y, self.hy, self.a, self.k, self.o, self.s, self.l = (
            X * PIXEL_MUL,
            y,
            hy,
            a,
            k,
            o,
            s,
            l,
        )


def clear_console() -> None:
    if name == "nt":
        _ = system("cls")
    else:
        _ = system("clear")


def round_list(l: list[float]) -> list[str]:
    return list(map(lambda x: f"{round(x, 3):.3f}", l))
