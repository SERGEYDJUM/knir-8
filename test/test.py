from cho.model import CHOss
from labeling.labeling import rawread, load_dataset, DATASET_RAWS
from sklearn import metrics
from cho import CHO
from os import path
import numpy as np


def main():
    df, _ = load_dataset()
    # df = df[df["human_score"] != -1]
    df = df[df["bbox_index"].isin((1, 2, 4, 5, 6, 7, 10))]
    roi_radius = 32
    shift_r = 4

    raws = {}

    X = np.zeros((df.shape[0], roi_radius * 2, roi_radius * 2), dtype=np.single)
    y = np.zeros(df.shape[0], dtype=np.bool)
    k = np.zeros(df.shape[0], dtype=np.bool)
    hy = np.zeros(df.shape[0], dtype=np.bool)

    for i, row in enumerate(df.itertuples()):
        raw_name = row.raw_source
        center = row.bbox_center_x, row.bbox_center_y
        if raw_name not in raws:
            raw_path = path.join(DATASET_RAWS, raw_name)
            max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
            raws[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

        center = (
            int(np.random.uniform(-shift_r, shift_r) + center[0]),
            int(np.random.uniform(-shift_r, shift_r) + center[1]),
        )

        X[i] = raws[raw_name][row.slice_index][
            center[1] - roi_radius : center[1] + roi_radius,
            center[0] - roi_radius : center[0] + roi_radius,
        ]

        y[i] = row.signal_present
        hy[i] = row.human_score
        k[i] = row.recon_kernel == "soft"

    X_train, y_train = X[np.logical_not(k)], y[np.logical_not(k)]
    # X_train, y_train = X[k], y[k]
    X_test, y_test = X[np.logical_not(k)], y[np.logical_not(k)]

    print(X_test.shape, X_train.shape)
    model = CHOss(channel_noise_std=1, test_stat_noise_std=1)
    measures = np.zeros(shape=128, dtype=np.float64)

    for i in range(measures.shape[0]):
        model.train(X_train, y_train)
        measures[i] = model.measure(X_test, y_test)

    print("Model AUC mean:", measures.mean())
    print("Model AUC std:", measures.std())

    # print("Human train AUC:", metrics.roc_auc_score(y_train, hy[k]))
    print("Human test AUC:", metrics.roc_auc_score(y_test, hy[np.logical_not(k)]))
