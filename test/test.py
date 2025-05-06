from typing import Self
from cho import CHO, CHOss, CHOArray, NPWMF
from model import CNNMO, RNMO

from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable, TableStyle
from scipy.stats import pearsonr
from numpy.typing import NDArray

from os import path
from sys import argv
from dataclasses import dataclass
from pandas import read_csv, DataFrame
from tqdm import tqdm

from .utils import DataStore, round_list, clear_console

import numpy as np
import torch


MEASURE_DIST_REPEAT = 128
NN_MEASURE_DIST_REPEAT = 128

CNNMO_CP_PATH: str = "checkpoints/cnn_mo.pt"
CNMMO_NOISE_STD: float = 2.85

RNMO_CP_PATH: str = "checkpoints/rn_mo.pt"
OUTPUT_PATH: str = "dataset/metrics.csv"

NPWMF_T_NOISE_STD = 0
NPWMF_TRAIN_SET_PART = 1

CHO_NOISE_MUL = 0.3
CHO_T_NOISE_STD = 2
CHO_TRAIN_SET_PART = 1

CHOSS_NOISE_MUL = 0.85
CHOSS_T_NOISE_STD = 1.3
CHOSS_TRAIN_SET_PART = 1

MO_RESTRICTED = "restrict" in argv

RNMO_ENABLED = False
MAIN_MODEL_NAME = "RN-MO" if RNMO_ENABLED else "CNN-MO"

ALT_MODEL = None
ALT_MODEL_NAME = None
ALT_MODEL_NAME_EXT = None
ALT_MODEL_NAME_FULL = None

if "npwmf" in argv:
    ALT_MODEL_NAME = "NPWMF"
    ALT_MODEL = NPWMF(
        test_stat_noise_std=NPWMF_T_NOISE_STD,
        train_set_keep=NPWMF_TRAIN_SET_PART,
    )
elif "cho" in argv:
    ALT_MODEL_NAME = "CHO"
    ALT_MODEL = CHO(
        channel_noise_std=CHO_NOISE_MUL,
        test_stat_noise_std=CHO_T_NOISE_STD,
        train_set_keep=CHO_TRAIN_SET_PART,
    )
else:
    ALT_MODEL_NAME = "CHOss"
    ALT_MODEL = CHOss(
        channel_noise_std=CHOSS_NOISE_MUL,
        test_stat_noise_std=CHOSS_T_NOISE_STD,
        train_set_keep=CHOSS_TRAIN_SET_PART,
    )


ALT_MODEL_NAME_EXT = ALT_MODEL_NAME + ("(r)" if MO_RESTRICTED else "")
ALT_MODEL_NAME_FULL = f"13x{ALT_MODEL_NAME}" + (
    " (restricted)" if MO_RESTRICTED else ""
)


@dataclass
class Results:
    human_aucs: list[float]
    main_aucs: list[float]
    main_auc_stds: list[float]
    alt_aucs: list[float]
    alt_auc_stds: list[float]

    def get_slice(self, start: int, end: int) -> Self:
        return Results(
            self.human_aucs[start:end],
            self.main_aucs[start:end],
            self.main_auc_stds[start:end],
            self.alt_aucs[start:end],
            self.alt_auc_stds[start:end],
        )

    def print_aucs(self, index_start: int = 1) -> None:
        table = PrettyTable()
        table.set_style(TableStyle.MARKDOWN)
        table.field_names = [
            "CFG",
            "Human",
            "CNN-MO (μ)",
            "CNN-MO (σ)",
            ALT_MODEL_NAME + " (μ)",
            ALT_MODEL_NAME + " (σ)",
        ]

        rows = zip(
            range(index_start, len(self.human_aucs) + index_start),
            round_list(self.human_aucs),
            round_list(self.main_aucs),
            round_list(self.main_auc_stds),
            round_list(self.alt_aucs),
            round_list(self.alt_auc_stds),
        )

        table.add_rows(list(rows), divider=True)
        print(table)

    def print_metrics(self) -> None:
        main_p, main_meandiff = adequacy_score(self.main_aucs, self.human_aucs)
        alt_p, alt_meandiff = adequacy_score(self.alt_aucs, self.human_aucs)

        table = PrettyTable()
        table.set_style(TableStyle.MARKDOWN)
        table.field_names = ["", "Δμ", "Pearson's ρ"]
        table.add_rows(
            [
                [
                    MAIN_MODEL_NAME,
                    f"{main_meandiff:.3f}",
                    f"{main_p:.3f}",
                ],
                [
                    ALT_MODEL_NAME,
                    f"{alt_meandiff:.3f}",
                    f"{alt_p:.3f}",
                ],
            ],
            divider=True,
        )
        print(table)


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

    for i in tqdm(range(n), desc=f"Sampling {ALT_MODEL_NAME} outputs", leave=False):
        if places is not None:
            measurements[i] = model.measure(X, y, disc=places)
        else:
            measurements[i] = model.measure(X, y)

    return measurements.mean(), measurements.std()


def measure_main(
    model: CNNMO,
    X: NDArray,
    y: NDArray,
    n: int = NN_MEASURE_DIST_REPEAT,
) -> float:
    measurements = np.zeros(n, dtype=np.double)

    for i in tqdm(range(n), desc=f"Sampling {MAIN_MODEL_NAME} outputs", leave=False):
        with torch.no_grad():
            inp = torch.from_numpy(X).reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
            out = model(inp).flatten()
            out += torch.normal(0, CNMMO_NOISE_STD, out.shape)
            measurements[i] = roc_auc_score(y, out)

    return measurements.mean(), measurements.std()


def adequacy_score(aucs: list[float], ref: list[float]) -> tuple[float]:
    n = len(aucs)
    assert n == len(ref)
    pear_r = pearsonr(aucs, ref).statistic
    deltamu = abs(sum(aucs) / n - sum(ref) / n)
    return pear_r, deltamu


class ExperimentExecutor:
    def __init__(self) -> None:
        self.data = DataStore()
        self.main_model = load_main_model()
        self.alt_model = ALT_MODEL

        print(f"{len(self.data.x)} images with metadata loaded.\n")

        self.results = Results([], [], [], [], [])

        self.kernel_soft = self.data.k
        self.small_objects = self.data.o
        self.kernel_standard = np.logical_not(self.kernel_soft)
        self.big_objects = np.logical_not(self.small_objects)
        self.scored = self.data.hy != -1
        self.shifted = self.data.l
        self.not_shifted = np.logical_not(self.shifted)

    def perform(
        self,
        is_small_objects: bool,
        is_kernel_soft: bool,
        tube_current: int,
        shifted: bool = False,
        n: int = None,
    ) -> None:
        current_f = self.data.a == tube_current
        object_size_f = self.small_objects if is_small_objects else self.big_objects
        kernel_f = self.kernel_soft if is_kernel_soft else self.kernel_standard
        shifted_f = self.shifted if shifted else self.not_shifted

        train_filter = np.logical_and(
            self.kernel_standard if MO_RESTRICTED else kernel_f,
            np.logical_and(
                self.not_shifted if MO_RESTRICTED else shifted_f,
                np.logical_and(object_size_f, current_f),
            ),
        )

        ex_filter = np.logical_and(
            np.logical_and(kernel_f, self.scored),
            np.logical_and(np.logical_and(current_f, object_size_f), shifted_f),
        )

        print(f"IQA CFG #{n}:")
        print(f"\t> Object size: {'4mm' if is_small_objects else '5mm'}")
        print(f"\t> Kernel type: {'soft' if is_kernel_soft else 'standard'}")
        print(f"\t> Tube current: {tube_current} mA")
        print(f"\t> Z-shifted: {shifted}")
        print(f"\t> Labeled images: {ex_filter.sum()}")

        alt_model = CHOArray(self.alt_model)

        print(
            f"\n\tTraining {ALT_MODEL_NAME_FULL} on {train_filter.sum()} images... ",
            end="",
        )

        alt_model.train(
            self.data.x[train_filter],
            self.data.y[train_filter],
            self.data.s[train_filter],
        )

        print("DONE\n")

        inp = self.data.x[ex_filter]
        gt = self.data.y[ex_filter]
        hy = self.data.hy[ex_filter]
        places = self.data.s[ex_filter] if isinstance(alt_model, CHOArray) else None

        assert inp.shape[0] == gt.shape[0] == hy.shape[0]

        human_auc = roc_auc_score(gt, hy)
        main_auc, main_auc_std = measure_main(self.main_model, inp, gt)
        alt_auc, alt_auc_std = measure_dist(alt_model, inp, gt, places=places)

        print("\tCalculated AUCs:")
        print(f"\t\tHuman Observer: {human_auc:.3f}")
        print(f"\t\t{MAIN_MODEL_NAME}: {main_auc:.3f} (std={main_auc_std:.4f})")
        print(f"\t\t{ALT_MODEL_NAME}: {alt_auc:.3f} (std={alt_auc_std:.4f})")
        print("\n")

        self.results.human_aucs.append(human_auc)
        self.results.main_aucs.append(main_auc)
        self.results.main_auc_stds.append(main_auc_std)
        self.results.alt_aucs.append(alt_auc)
        self.results.alt_auc_stds.append(alt_auc_std)


def tune():
    ex = ExperimentExecutor()

    with open(".temp/tuning.csv", "w") as tfile:
        print(f"Tuning {ALT_MODEL_NAME_FULL}")
        print("noise_std,r,md", file=tfile)
        for noise_std in np.arange(0.95, 1.32, 0.01):
            ex.results = Results([], [], [], [], [])
            ex.alt_model.ch_noise_std = noise_std

            ex.perform(
                is_small_objects=True,
                is_kernel_soft=False,
                tube_current=10,
            )

            ex.perform(
                is_small_objects=True,
                is_kernel_soft=False,
                tube_current=40,
            )

            ex.perform(
                is_small_objects=False,
                is_kernel_soft=False,
                tube_current=10,
            )

            ex.perform(
                is_small_objects=False,
                is_kernel_soft=False,
                tube_current=40,
            )

            ade_alt, alt_meandiff = adequacy_score(
                ex.results.alt_aucs, ex.results.human_aucs
            )
            print(f"{noise_std:.4f},{ade_alt:.5f},{alt_meandiff:.3f}")
            print(f"{noise_std:.4f},{ade_alt:.5f},{alt_meandiff:.3f}", file=tfile)

            exit()


def main():
    if "tune" in argv:
        tune()

    ex = ExperimentExecutor()

    # Configuration #1
    ex.perform(
        n=1,
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #2
    ex.perform(
        n=2,
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=40,
    )

    # Configuration #3
    ex.perform(
        n=3,
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=10,
    )

    # Configuration #4
    ex.perform(
        n=4,
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=40,
    )

    print()
    ex.results.print_aucs()
    print()
    ex.results.print_metrics()
    print()

    # Configuration #5
    ex.perform(
        n=5,
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #6
    ex.perform(
        n=6,
        is_small_objects=True,
        is_kernel_soft=True,
        tube_current=40,
    )

    # Configuration #7
    ex.perform(
        n=7,
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=10,
    )

    # Configuration #8
    ex.perform(
        n=8,
        is_small_objects=False,
        is_kernel_soft=True,
        tube_current=40,
    )

    # Configuration #9
    ex.perform(
        n=9,
        is_small_objects=True,
        is_kernel_soft=False,
        tube_current=40,
        shifted=True,
    )

    # Configuration #10
    ex.perform(
        n=10,
        is_small_objects=False,
        is_kernel_soft=False,
        tube_current=10,
        shifted=True,
    )

    clear_console()
    print(f"# {MAIN_MODEL_NAME} vs {ALT_MODEL_NAME_FULL}")
    print("\n## Train set AUCs\n")
    res_train = ex.results.get_slice(0, 4)
    res_train.print_aucs()
    print("\n### Metrics\n")
    res_train.print_metrics()

    print("\n## Test set AUCs\n")
    res_test = ex.results.get_slice(4, 10)
    res_test.print_aucs(5)
    print("\n### Metrics\n")
    res_test.print_metrics()

    res_all = ex.results.get_slice(0, 10)
    print("\n## Full dataset metrics\n")
    res_all.print_metrics()

    df = DataFrame()

    if path.exists(OUTPUT_PATH):
        df = read_csv(OUTPUT_PATH)

    df["CFG"] = list(range(len(res_all.human_aucs)))
    df["Human"] = res_all.human_aucs
    df[MAIN_MODEL_NAME] = res_all.main_aucs
    df[MAIN_MODEL_NAME + "_std"] = res_all.main_auc_stds
    df[ALT_MODEL_NAME_EXT] = res_all.alt_aucs
    df[ALT_MODEL_NAME_EXT + "_std"] = res_all.alt_auc_stds

    df.to_csv(OUTPUT_PATH, index=False)
