from labeling.labeling import rawread, load_dataset, DATASET_RAWS
from sklearn import metrics
from cho import CHO
from os import path
import numpy as np


def main():
    df, _ = load_dataset()
    df = df[df["human_score"] != -1]
    roi_radius = 32
    shift_r = 4

    raws = {}

    X = np.zeros((df.shape[0], roi_radius * 2, roi_radius * 2), dtype=np.single)
    y = np.zeros(df.shape[0], dtype=np.bool)
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

    model = CHO(channel_noise_std=4, test_stat_noise_std=12, _debug_mode=True)
    # model.train(X[0::2, :, :], y[0::2])
    model.train(X, y)

    # print("Template:", model.template)
    print("Template sum:", model.template.sum())

    measures = []
    measures_cnt = 100
    # X_test = X[1::2]
    # y_test = y[1::2]
    X_test = X
    y_test = y

    for i in range(measures_cnt):
        measures.append(model.measure(X_test, y_test))

    print("Model AUC:", sum(measures) / measures_cnt)
    print("Human AUC:", metrics.roc_auc_score(y, hy))
