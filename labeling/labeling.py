import struct
from tkinter import *
from tkinter import ttk
from os import path
from sys import argv
from pandas import DataFrame, read_csv
from random import shuffle
from PIL import Image, ImageTk, ImageOps
from numpy.typing import NDArray
import numpy as np

DATASET_DIR = path.abspath("./dataset")
DATASET_RAWS = path.join(DATASET_DIR, "raws")
DATASET_CSV = path.join(DATASET_DIR, "dataset.csv")

image_cache: dict[str, NDArray] = {}


def rawread(path: str, shape: tuple) -> NDArray:
    with open(path, "rb") as fin:
        data = fin.read()

    fmt = ["f", 4, np.single]
    data = struct.unpack("%d%s" % (len(data) / fmt[1], fmt[0]), data)
    data = np.array(data, dtype=fmt[2])

    if shape:
        data = data.reshape(shape)

    return data


def load_dataset(raw_filter: list[str] = None) -> tuple[DataFrame, list[int]]:
    csvdf = read_csv(DATASET_CSV)
    unscored_indexes = []

    if raw_filter:
        isin = csvdf["raw_source"].isin(raw_filter)
        hs = csvdf["human_score"] == -1
        unscored_indexes = csvdf.index[isin * hs].tolist()
    else:
        unscored_indexes = csvdf.index[csvdf["human_score"] == -1].tolist()

    shuffle(unscored_indexes)
    shuffle(unscored_indexes)

    return csvdf, unscored_indexes


def load_image(df: DataFrame, row_idx: int) -> NDArray:
    global image_cache

    row = df.iloc[row_idx]
    raw_name = row["raw_source"]
    center = row["bbox_center_x"], row["bbox_center_y"]
    radius = int(max(row["bbox_safe_r_x"], row["bbox_safe_r_y"]))

    if raw_name not in image_cache:
        raw_path = path.join(DATASET_RAWS, raw_name)
        max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
        image_cache[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

    tomo_slice = image_cache[raw_name][row["slice_index"]]

    # center_shift = np.random.uniform(-radius / 5, radius / 5, 2)
    # radius = int(radius * 4 / 5)

    # center = (
    #     int(center[0] + center_shift[0]),
    #     int(center[1] + center_shift[1]),
    # )

    image = tomo_slice[
        max(center[1] - radius, 0) : center[1] + radius,
        max(center[0] - radius, 0) : center[0] + radius,
    ].copy()

    if np.random.uniform(0, 1) > 0.5:
        image = np.fliplr(image)

    if np.random.uniform(0, 1) > 0.5:
        image = np.flipud(image)

    print(
        f"Brightness distribution: {image.min():.1f}:{image.max():.1f}, Avg: {image.mean():.1f}\n"
    )
    image = np.clip(image, -1000, 1000)
    image -= np.min(image)
    image *= 255 / np.max(image)
    image = np.astype(image, np.uint8)

    assert image.shape[0] == image.shape[1]
    return image


def image_to_tk(imgarr: NDArray, scale: int = 5) -> ImageTk:
    image = np.kron(imgarr, np.ones((scale, scale)))
    image = Image.fromarray(image)

    return ImageTk.PhotoImage(image)


class App:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.wm_resizable(False, False)
        self.window.title("CT Labeling")
        self.init_complete = False

        self.csvdf, self.unscored_idxs = load_dataset(raw_filter=argv[1:])
        self.current_index = 0

        if not self.unscored_idxs:
            print("No images to label")
            exit(0)

        self.next(reverse=True)

        self.img_label = ttk.Label(self.window, image=self.image_m)
        self.img_label.grid(row=0, column=0)

        # self.img_label_s = ttk.Label(self.window, image=self.image_s)
        # self.img_label_s.grid(row=0, column=1)

        self.window.bind("z", lambda _: self.mark(False))
        self.window.bind("/", lambda _: self.mark(True))

        # self.button_frame = ttk.Frame(self.window)
        # self.button_frame.grid(row=0, column=2)

        # self.button_back = ttk.Button(
        #     self.button_frame, text="Предыдущее", command=lambda: self.next(True)
        # )
        # self.button_back.grid(row=0, column=0)

        # self.button_pos = ttk.Button(
        #     self.button_frame, text="Есть сигнал", command=lambda: self.mark(True)
        # )
        # self.button_pos.grid(row=1, column=0)

        # self.button_neg = ttk.Button(
        #     self.button_frame, text="Нет сигнала", command=lambda: self.mark(False)
        # )
        # self.button_neg.grid(row=1, column=1)

        self.init_complete = True

    def next(self, reverse: bool = False):
        if reverse:
            self.current_index = max(0, self.current_index - 1)
        else:
            if self.current_index == len(self.unscored_idxs) - 1:
                self.on_shutdown()
                exit(0)

            self.current_index = min(
                len(self.unscored_idxs) - 1, self.current_index + 1
            )

        image = load_image(self.csvdf, self.unscored_idxs[self.current_index])

        # self.image_s = image_to_tk(image, scale=2)
        self.image_m = image_to_tk(image, scale=4)

        if self.init_complete:
            # self.img_label_s.config(image=self.image_s)
            self.img_label.config(
                image=self.image_m,
                padding=(
                    np.random.uniform(0, 200),
                    np.random.uniform(0, 200),
                    100,
                    100,
                ),
            )

    def mark(self, present: bool):
        assert present is not None
        idx = self.unscored_idxs[self.current_index]

        if present:
            self.csvdf.at[idx, "human_score"] = 1
        else:
            self.csvdf.at[idx, "human_score"] = 0

        self.csvdf.to_csv(DATASET_CSV, index=False)
        self.next()

    def on_shutdown(self):
        matched_cnt = 0
        unmatched_cnt = 0

        for idx in self.unscored_idxs:
            if self.csvdf.at[idx, "human_score"] == -1:
                continue

            if (
                bool(self.csvdf.at[idx, "human_score"])
                == self.csvdf.at[idx, "signal_present"]
            ):
                matched_cnt += 1
            else:
                unmatched_cnt += 1

        all_cnt = matched_cnt + unmatched_cnt

        if all_cnt != 0:
            print(f"Scored {all_cnt} images.")
            print(f"Session precision: {matched_cnt * 100 / all_cnt:.1f}%.")

    def run(self):
        self.window.mainloop()
        self.on_shutdown()
