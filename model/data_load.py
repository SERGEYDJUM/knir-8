from os import path
from torch.utils.data import Dataset
import pandas as pd
import torch


from labeling.labeling import rawread, load_dataset, DATASET_RAWS

BLACKLISTED_CFGS = [
    "cd7e0f84b35facac6a17c69a697754d2",
    "94de7be616c1e06a7456341593426681",
]


class MyDataset(Dataset):
    def __init__(
        self,
        img_r: int = 32,
        train: bool = True,
        train_split: float = 0.9,
        value_scale: float = 0.001,
        extra_roi_mult: float = 1.1,
        random_state: int = 1,
        allowed_kernel: str = "standard",
    ) -> None:
        df, _ = load_dataset()
        df = df[df["human_score"] != -1]
        df = df[df["recon_kernel"] == allowed_kernel]
        df = df[~(df["xcist_cfg_md5"].isin(BLACKLISTED_CFGS))]

        raw_img_cache = dict()

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source

            if raw_name not in raw_img_cache:
                raw_path = path.join(DATASET_RAWS, raw_name)
                max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
                raw_img_cache[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        N = int((df.shape[0] * (train_split if train else (1 - train_split))) / 2)
        df_p = df[df["signal_present"] == True].reset_index(drop=True)
        df_n = df[df["signal_present"] == False].reset_index(drop=True)

        if train:
            df = pd.concat([df_p[:N], df_n[:N]], ignore_index=True)
        else:
            df = pd.concat([df_p[-N:], df_n[-N:]], ignore_index=True)

        augments = 8
        N *= 2 * augments
        # sroi_r = int(img_r * extra_roi_mult)

        self.X = torch.zeros((N, 1, img_r * 2, img_r * 2), dtype=torch.float32)
        self.y = torch.zeros((N, 1), dtype=torch.float32)
        self.gt = torch.zeros((N, 1), dtype=torch.float32)

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source
            center = row.bbox_center_x, row.bbox_center_y

            for j in range(augments):
                img = torch.from_numpy(
                    raw_img_cache[raw_name][row.slice_index][
                        # center[1] - sroi_r : center[1] + sroi_r,
                        center[1] - img_r : center[1] + img_r,
                        # center[0] - sroi_r : center[0] + sroi_r,
                        center[0] - img_r : center[0] + img_r,
                    ]
                )

                if j % 2 == 1:
                    img = torch.fliplr(img)

                if j in [2, 3, 6, 7]:
                    img = torch.flipud(img)

                if j > 3:
                    img = torch.rot90(img)

                self.X[i * augments + j, 0] = img * value_scale
                self.y[i * augments + j, 0] = row.human_score
                self.gt[i * augments + j, 0] = 1.0 if row.signal_present else 0.0

        del raw_img_cache

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple:
        return (self.X[idx], self.y[idx], self.gt[idx])

    def __getitems__(self, indices: list[int]) -> list[tuple]:
        return list(map(lambda idx: (self.X[idx], self.y[idx], self.gt[idx]), indices))
