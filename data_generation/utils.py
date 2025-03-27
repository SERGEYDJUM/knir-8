from dataclasses import dataclass
from typing import Self
from numpy.typing import NDArray
import numpy as np
import json


def save_raw(data: NDArray, path: str):
    with open(path, "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(data))


class LesionBBox:
    def __init__(
        self,
        center: tuple[int, int] | tuple[int, int, int],
        r: int | tuple[int, int] | tuple[int, int, int],
        safe_r: int | tuple[int, int] | tuple[int, int, int] = None,
        roi_r: int | tuple[int, int] | tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.center = center
        self.r = r
        self.sr = safe_r
        self.rr = roi_r

        if isinstance(r, int):
            self.r = (r, r, r)

        if isinstance(safe_r, int):
            self.sr = (safe_r, safe_r, safe_r)

        if isinstance(roi_r, int):
            self.rr = (roi_r, roi_r, roi_r)

    def _bbox_2d(self, r: tuple) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            self.center[0] - r[0],
            self.center[1] - r[1],
        ), (
            self.center[0] + r[0],
            self.center[1] + r[1],
        )

    def _bbox_3d(self, r: tuple) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (
            self.center[0] - r[0],
            self.center[1] - r[1],
            self.center[2] - r[2],
        ), (
            self.center[0] + r[0],
            self.center[1] + r[1],
            self.center[2] + r[2],
        )

    def bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.r) if include_z else self._bbox_2d(self.r)

    def safe_bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.sr) if include_z else self._bbox_2d(self.sr)

    def roi_bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.rr) if include_z else self._bbox_2d(self.rr)

    def scale(
        self, scale_factor: float, old_im_center: tuple[int], new_im_center: tuple[int]
    ):
        def scaler(x: int) -> int:
            return int(x * scale_factor)

        def retuple(t: tuple) -> tuple:
            return tuple(map(scaler, t))

        def recenter(t: tuple) -> tuple:
            shifted = map(lambda p: p[1] - p[0], zip(old_im_center, self.center))
            scaled = retuple(tuple(shifted))
            return tuple(map(lambda p: p[1] + p[0], zip(new_im_center, scaled)))

        return LesionBBox(
            center=recenter(self.center),
            r=retuple(self.r),
            safe_r=retuple(self.sr),
            roi_r=retuple(self.rr),
        )


@dataclass
class Phantom:
    signals: list[LesionBBox]

    def transform(
        self, scale: float, img_center: tuple[int], new_img_center: tuple[int]
    ) -> None:
        self.signals = [
            bbox.scale(scale, img_center, new_img_center) for bbox in self.signals
        ]

    def dump(self, path: str) -> None:
        bboxes = list(
            map(
                lambda b: {"center": b.center, "r": b.r, "rr": b.rr, "sr": b.sr},
                self.signals,
            )
        )

        with open(path, "w", encoding="utf-8") as phantom_file:
            json.dump(bboxes, fp=phantom_file)

    @staticmethod
    def load(path: str) -> Self:
        with open(path, "r", encoding="utf-8") as phantom_file:
            signals = json.load(phantom_file)
            return Phantom(
                signals=list(
                    map(
                        lambda b: LesionBBox(
                            center=tuple(b["center"]),
                            r=tuple(b["r"]),
                            safe_r=tuple(b["sr"]),
                            roi_r=tuple(b["rr"]),
                        ),
                        signals,
                    )
                )
            )
