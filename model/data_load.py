from os import path
from torchvision.transforms.v2 import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torch.utils.data import Dataset
import pandas as pd
import torch


from labeling.labeling import rawread, load_dataset, DATASET_RAWS


class MyDataset(Dataset):
    def __init__(
        self,
        img_r: int = 32,
        train: bool = True,
        train_split: float = 0.85,
        augments: int = 4,
        value_scale: float = 0.001,
        extra_roi_mult: float = 1.32,
    ) -> None:
        self.transforms = Compose(
            [
                # RandomCrop(img_r * 2),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

        df, _ = load_dataset()

        assert len(df[df["human_score"] == -1]) == 0

        raw_img_cache = dict()

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source

            if raw_name not in raw_img_cache:
                raw_path = path.join(DATASET_RAWS, raw_name)
                max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
                raw_img_cache[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

        N = int((df.shape[0] * (train_split if train else (1 - train_split))) / 2)
        df_p = df[df["signal_present"] == True].reset_index(drop=True)
        df_n = df[df["signal_present"] == False].reset_index(drop=True)

        if train:
            df = pd.concat([df_p[:N], df_n[:N]], ignore_index=True)
        else:
            df = pd.concat([df_p[-N:], df_n[-N:]], ignore_index=True)

        N *= 2 * augments
        sroi_r = int(img_r * extra_roi_mult)

        self.X = torch.zeros((N, 1, img_r * 2, img_r * 2), dtype=torch.float32)
        self.y = torch.zeros((N, 1), dtype=torch.float32)
        self.gt = torch.zeros((N, 1), dtype=torch.float32)

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source
            center = row.bbox_center_x, row.bbox_center_y

            for j in range(augments):
                cropped = self.transforms(
                    torch.from_numpy(
                        raw_img_cache[raw_name][row.slice_index][
                            # center[1] - sroi_r : center[1] + sroi_r,
                            center[1] - img_r : center[1] + img_r,
                            # center[0] - sroi_r : center[0] + sroi_r,
                            center[0] - img_r : center[0] + img_r,
                        ]
                    )
                )

                self.X[i * augments + j, 0] = cropped * value_scale
                self.y[i * augments + j, 0] = row.human_score
                self.gt[i * augments + j, 0] = 1.0 if row.signal_present else 0.0

        del raw_img_cache

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple:
        return (self.X[idx], self.y[idx], self.gt[idx])

    def __getitems__(self, indices: list[int]) -> list[tuple]:
        return list(map(lambda idx: (self.X[idx], self.y[idx], self.gt[idx]), indices))
