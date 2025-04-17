from labeling.labeling import rawread, load_dataset, DATASET_RAWS
from cho import CHO, CHOss, CHOArray
from model import CNNMO, RNMO

from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable, TableStyle
from scipy.stats import pearsonr
from numpy.typing import NDArray

from os import path
from sys import argv
from dataclasses import dataclass
from pandas import read_csv, DataFrame

import numpy as np
import torch


ROI_R: int = 32
PIXEL_MUL = 0.001

MEASURE_DIST_REPEAT = 256

CNNMO_CP_PATH: str = "checkpoints/cnn_mo.pt"
RNMO_CP_PATH: str = "checkpoints/rn_mo.pt"
OUTPUT_PATH: str = "dataset/metrics.csv"

CHO_NOISE_MUL = 1.4
CHO_T_NOISE_STD = 0
CHO_TRAIN_SET_PART = 1

CHOSS_NOISE_MUL = 0.85
CHOSS_T_NOISE_STD = 0.5
CHOSS_TRAIN_SET_PART = 1

CHOSS_ENABLED = "choss" in argv
CHO_RESTRICTED = "restrict" in argv

RNMO_ENABLED = False

MAIN_MODEL_NAME = "RN-MO" if RNMO_ENABLED else "CNN-MO"
ALT_MODEL_NAME = "CHOss" if CHOSS_ENABLED else "CHO"
ALT_MODEL_NAME_EXT = ALT_MODEL_NAME + ("(r)" if CHO_RESTRICTED else "")
ALT_MODEL_NAME_FULL = "13xCHOss" if CHOSS_ENABLED else "13xCHO"
ALT_MODEL_NAME_FULL += " (restricted)" if CHO_RESTRICTED else ""


@dataclass
class Results:
    human_aucs: list[float]
    main_aucs: list[float]
    alt_aucs: list[float]
    alt_auc_stds: list[float]


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
    checkpoint = torch.load(RNMO_CP_PATH if RNMO_ENABLED else CNNMO_CP_PATH)

    model = RNMO() if RNMO_ENABLED else CNNMO()
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def measure_dist(
    model: CHO | CHOss | CHOArray,
    X: NDArray,
    y: NDArray,
    n: int = MEASURE_DIST_REPEAT,
    places: NDArray = None,
) -> tuple[float, float]:
    measurements = np.zeros(n, dtype=np.double)

    for i in range(n):
        if places is not None:
            measurements[i] = model.measure(X, y, disc=places)
        else:
            measurements[i] = model.measure(X, y)

    return measurements.mean(), measurements.std()


def measure_main(model: CNNMO, X: NDArray, y: NDArray) -> float:
    with torch.no_grad():
        inp = torch.from_numpy(X).reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        return roc_auc_score(y, model(inp).flatten())


def calc_corrs(
    human_aucs: list, main_aucs: list, alt_aucs: list
) -> tuple[tuple, tuple]:
    main_sc_p = pearsonr(human_aucs, main_aucs).statistic
    alt_sc_p = pearsonr(human_aucs, alt_aucs).statistic
    return main_sc_p, alt_sc_p


def adequacy_score(aucs: list[float], ref: list[float]) -> tuple[float]:
    n = len(aucs)
    assert n == len(ref)
    # mse = sum(map(lambda t: (t[0] - t[1]) ** 2, zip(aucs, ref))) / len(ref)
    pear_r = pearsonr(aucs, ref).statistic
    deltamu = abs(sum(aucs) / n - sum(ref) / n)
    return pear_r, deltamu


def round_list(l: list[float]) -> list[str]:
    return list(map(lambda x: f"{round(x, 3):.3f}", l))


def print_auc_table(
    human_aucs: list[float],
    main_aucs: list[float],
    alt_aucs: list[float],
    alt_auc_stds: list[float],
    index_start: int = 1,
) -> None:
    table = PrettyTable()
    table.set_style(TableStyle.MARKDOWN)
    table.field_names = [
        "CFG",
        "Human",
        "CNN-MO",
        ALT_MODEL_NAME + " (μ)",
        ALT_MODEL_NAME + " (σ)",
    ]
    table.add_rows(
        list(
            zip(
                range(index_start, len(human_aucs) + index_start),
                round_list(human_aucs),
                round_list(main_aucs),
                round_list(alt_aucs),
                round_list(alt_auc_stds),
            )
        ),
        divider=True,
    )

    print(table)


def print_res_table(
    main_p: float,
    alt_p: float,
    main_mse: float,
    alt_mse: float,
    main_adequacy: float,
    alt_adequacy: float,
) -> None:
    table = PrettyTable()
    table.set_style(TableStyle.MARKDOWN)
    table.field_names = ["", "Pearson's ρ", "Δμ", "Adequacy"]
    table.add_rows(
        [
            [
                MAIN_MODEL_NAME,
                f"{main_p:.3f}",
                f"{main_mse:.3f}",
                f"__{main_adequacy:.3f}__",
            ],
            [
                ALT_MODEL_NAME,
                f"{alt_p:.3f}",
                f"{alt_mse:.3f}",
                f"__{alt_adequacy:.3f}__",
            ],
        ],
        divider=True,
    )

    print(table)


class ExperimentExecutor:
    def __init__(self) -> None:
        self.data = DataStore()
        self.main_model = load_main_model()
        self.results = Results([], [], [], [])

        self.kernel_soft = self.data.k
        self.small_objects = self.data.o
        self.kernel_standard = np.logical_not(self.kernel_soft)
        self.big_objects = np.logical_not(self.small_objects)
        self.scored = self.data.hy != -1

    def perform(
        self, is_small_objects: bool, is_kernel_soft: bool, tube_current: int
    ) -> None:
        current_f = self.data.a == tube_current
        object_size_f = self.small_objects if is_small_objects else self.big_objects
        kernel_f = self.kernel_soft if is_kernel_soft else self.kernel_standard

        alt_model = CHOArray(
            CHOss(
                channel_noise_std=CHOSS_NOISE_MUL,
                test_stat_noise_std=CHOSS_T_NOISE_STD,
                train_set_keep=CHOSS_TRAIN_SET_PART,
            )
            if CHOSS_ENABLED
            else CHO(
                channel_noise_std=CHO_NOISE_MUL,
                test_stat_noise_std=CHO_T_NOISE_STD,
                train_set_keep=CHO_TRAIN_SET_PART,
            )
        )

        train_filter = np.logical_and(
            self.kernel_standard if CHO_RESTRICTED else kernel_f,
            np.logical_and(current_f, object_size_f),
        )

        alt_model.train(
            self.data.x[train_filter] * PIXEL_MUL,
            self.data.y[train_filter],
            self.data.s[train_filter],
        )

        ex_filter = np.logical_and(
            np.logical_and(kernel_f, self.scored),
            np.logical_and(current_f, object_size_f),
        )

        inp = self.data.x[ex_filter] * PIXEL_MUL
        gt = self.data.y[ex_filter]
        hy = self.data.hy[ex_filter]
        places = self.data.s[ex_filter] if isinstance(alt_model, CHOArray) else None

        assert inp.shape[0] == gt.shape[0] == hy.shape[0]

        human_auc = roc_auc_score(gt, hy)
        main_auc = measure_main(self.main_model, inp, gt)
        alt_auc, alt_auc_std = measure_dist(alt_model, inp, gt, places=places)

        self.results.human_aucs.append(human_auc)
        self.results.main_aucs.append(main_auc)
        self.results.alt_aucs.append(alt_auc)
        self.results.alt_auc_stds.append(alt_auc_std)


def main():
    ex = ExperimentExecutor()

    print(f"# {MAIN_MODEL_NAME} vs {ALT_MODEL_NAME_FULL}")

    # Configuration #1
    ex.perform(
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #2
    ex.perform(
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=40,
    )

    # Configuration #3
    ex.perform(
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #4
    ex.perform(
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=40,
    )

    print("\n## Train set AUCs\n")
    res = ex.results
    main_p, alt_p = calc_corrs(res.human_aucs, res.main_aucs, res.alt_aucs)
    ade_main, main_mse = adequacy_score(res.main_aucs, res.human_aucs)
    ade_alt, alt_mse = adequacy_score(res.alt_aucs, res.human_aucs)
    print_auc_table(res.human_aucs, res.main_aucs, res.alt_aucs, res.alt_auc_stds)
    print("\n### Metrics\n")
    print_res_table(main_p, alt_p, main_mse, alt_mse, ade_main, ade_alt)

    # Configuration #5
    ex.perform(
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #6
    ex.perform(
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=40,
    )

    # Configuration #7
    ex.perform(
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #8
    ex.perform(
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=40,
    )

    print("\n## Test set AUCs\n")
    res = ex.results
    print_auc_table(
        res.human_aucs[4:],
        res.main_aucs[4:],
        res.alt_aucs[4:],
        res.alt_auc_stds[4:],
        index_start=5,
    )
    main_p, alt_p = calc_corrs(res.human_aucs[4:], res.main_aucs[4:], res.alt_aucs[4:])
    ade_main, main_mse = adequacy_score(res.main_aucs[4:], res.human_aucs[4:])
    ade_alt, alt_mse = adequacy_score(res.alt_aucs[4:], res.human_aucs[4:])
    print("\n### Metrics\n")
    print_res_table(main_p, alt_p, main_mse, alt_mse, ade_main, ade_alt)

    print("\n## Dataset metrics\n")
    main_p, alt_p = calc_corrs(res.human_aucs, res.main_aucs, res.alt_aucs)
    ade_main, main_mse = adequacy_score(res.main_aucs, res.human_aucs)
    ade_alt, alt_mse = adequacy_score(res.alt_aucs, res.human_aucs)
    print_res_table(main_p, alt_p, main_mse, alt_mse, ade_main, ade_alt)

    df = DataFrame()
    if path.exists(OUTPUT_PATH):
        df = read_csv(OUTPUT_PATH)
    df["CFG"] = list(range(len(res.human_aucs)))
    df["Human"] = res.human_aucs
    df[MAIN_MODEL_NAME] = res.main_aucs
    df[ALT_MODEL_NAME_EXT] = res.alt_aucs
    df[ALT_MODEL_NAME_EXT + "_std"] = res.alt_auc_stds
    df.to_csv(OUTPUT_PATH, index=False)
