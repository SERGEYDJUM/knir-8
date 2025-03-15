from os import path
from torchvision.transforms.v2 import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torch.utils.data import Dataset
import torch


from labeling.labeling import rawread, load_dataset, DATASET_RAWS


class MyDataset(Dataset):
    def __init__(
        self,
        img_r: int = 32,
        train: bool = True,
        train_split: float = 0.75,
        augments: int = 4,
    ) -> None:
        self.transforms = Compose(
            [
                RandomCrop(img_r * 2),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

        df, _ = load_dataset()
        df = df.sample(frac=1, random_state=1)

        if train:
            df = df.head(int(df.shape[0] * train_split))
        else:
            df = df.tail(int(df.shape[0] * (1 - train_split)))

        N = df.shape[0] * augments
        sroi_r = img_r * 3 // 2

        self.X = torch.zeros((N, 1, img_r * 2, img_r * 2), dtype=torch.float32)
        self.y = torch.zeros((N, 1), dtype=torch.float32)

        raw_img_cache = {}

        for i, row in enumerate(df.itertuples()):
            raw_name = row.raw_source
            center = row.bbox_center_x, row.bbox_center_y

            if raw_name not in raw_img_cache:
                raw_path = path.join(DATASET_RAWS, raw_name)
                max_slice = df[df["raw_source"] == raw_name]["slice_index"].max()
                raw_img_cache[raw_name] = rawread(raw_path, (max_slice + 1, 512, 512))

            for j in range(augments):
                self.y[i + j, 0] = float(row.signal_present)
                self.X[i + j, 0] = self.transforms(
                    torch.from_numpy(
                        raw_img_cache[raw_name][row.slice_index][
                            center[1] - sroi_r : center[1] + sroi_r,
                            center[0] - sroi_r : center[0] + sroi_r,
                        ]
                    )
                )

        del raw_img_cache

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple:
        return (self.X[idx], self.y[idx])

    def __getitems__(self, indices: list[int]) -> list[tuple]:
        return list(map(lambda idx: (self.X[idx], self.y[idx]), indices))
