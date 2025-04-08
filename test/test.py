from labeling.labeling import rawread, load_dataset, DATASET_RAWS
from cho import CHO, CHOss
from model import CNNMO

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from numpy.typing import NDArray

from os import path

import numpy as np
import torch


ROI_R: int = 32
CNNMO_CP_PATH: str = "checkpoints/cnn_mo.pt"
CNNMO_INP_PIXEL_MUL = 0.001


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
        assert len(df[df["human_score"] == -1]) == 0

        raws = {}

        X = np.zeros((df.shape[0], ROI_R * 2, ROI_R * 2), dtype=np.single)
        y = np.zeros(df.shape[0], dtype=np.bool)
        k = np.zeros(df.shape[0], dtype=np.bool)
        a = np.zeros(df.shape[0], dtype=np.uint8)
        hy = np.zeros(df.shape[0], dtype=np.bool)
        o = np.zeros(df.shape[0], dtype=np.bool)
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
            o[i] = row.phantom_cfg_md5 == "3d3fd108138ce807fecba23851bbf61b"
            s[i] = row.bbox_index

            center = row.bbox_center_x, row.bbox_center_y
            X[i] = raws[raw_name][row.slice_index][
                center[1] - ROI_R : center[1] + ROI_R,
                center[0] - ROI_R : center[0] + ROI_R,
            ]

        self.x, self.y, self.hy, self.a, self.k, self.o, self.s = X, y, hy, a, k, o, s


def load_main_model() -> CNNMO:
    checkpoint = torch.load(CNNMO_CP_PATH)

    model = CNNMO()
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def measure_dist(
    model: CHO | CHOss, X: NDArray, y: NDArray, n: int = 128
) -> tuple[float, float]:
    measurements = np.zeros(n, dtype=np.double)

    for i in range(n):
        measurements[i] = model.measure(X, y)

    return measurements.mean(), measurements.std()


def measure_main(model: CNNMO, X: NDArray, y: NDArray) -> float:
    with torch.no_grad():
        inp = torch.from_numpy(X * CNNMO_INP_PIXEL_MUL)
        inp = inp.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        return roc_auc_score(y, model(inp).flatten())


def main():
    data = DataStore()

    kernel_soft = data.k
    kernel_standard = np.logical_not(kernel_soft)

    small_objects = data.o
    big_objects = np.logical_not(small_objects)

    main_model = load_main_model()
    alt_model = CHOss(channel_noise_std=1, test_stat_noise_std=1)

    human_aucs = []
    main_aucs = []
    alt_aucs = []
    alt_auc_stds = []

    def perform_experiment(
        is_small_objects: bool, is_kernel_soft: bool, tube_current: int
    ) -> None:
        current_f = data.a == tube_current
        object_size_f = small_objects if is_small_objects else big_objects
        kernel_f = kernel_soft if is_kernel_soft else kernel_standard

        ex_filter = np.logical_and(
            kernel_f,
            np.logical_and(current_f, object_size_f),
        )

        inp = data.x[ex_filter]
        gt = data.y[ex_filter]
        hy = data.hy[ex_filter]

        assert inp.shape[0] == gt.shape[0] == hy.shape[0]

        alt_model.train(inp, gt)

        human_auc = roc_auc_score(gt, hy)
        main_auc = measure_main(main_model, inp, gt)
        alt_auc, alt_auc_std = measure_dist(alt_model, inp, gt)

        print(
            f"Human={human_auc:.3f}, Main={main_auc:.3f}, Alt={alt_auc:.3f} (Alt Std={alt_auc_std:.3f})"
        )

        human_aucs.append(human_auc)
        main_aucs.append(main_auc)
        alt_aucs.append(alt_auc)
        alt_auc_stds.append(alt_auc_std)

    # Configuration #1
    perform_experiment(
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #2
    perform_experiment(
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=40,
    )

    # Configuration #3
    perform_experiment(
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #4
    perform_experiment(
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=40,
    )

    # Configuration #5
    perform_experiment(
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #6
    perform_experiment(
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=40,
    )

    # Configuration #7
    perform_experiment(
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #8
    perform_experiment(
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=40,
    )

    main_sc = spearmanr(human_aucs, main_aucs)
    alt_sc = spearmanr(human_aucs, alt_aucs)

    print(
        f"Spearman correlation: Main={main_sc.statistic:.3f}, Alt={alt_sc.statistic:.3f}"
    )

    main_sc = pearsonr(human_aucs, main_aucs)
    alt_sc = pearsonr(human_aucs, alt_aucs)

    print(
        f"Pearson correlation: Main={main_sc.statistic:.3f}, Alt={alt_sc.statistic:.3f}"
    )
