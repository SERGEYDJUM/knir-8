from itertools import product
from dataclasses import dataclass
from os import makedirs

from numpy.typing import NDArray
from sklearn import metrics
from PIL import Image
import numpy as np


def _gabor_filter(x: int, y: int, f: float, w: float, theta: float) -> float:
    exp_component = -4 * np.log(2) * (x * x + y * y) / (w * w)
    cos_component = 2 * np.pi * f * (x * np.cos(theta) + y * np.sin(theta))
    return np.exp(exp_component) * np.cos(cos_component)


@dataclass
class CHOChannelConfig:
    bands: tuple[tuple[float, float]] = (
        (3 / 256, 112.96),
        (3 / 128, 56.48),
        (3 / 64, 28.24),
        (3 / 32, 14.12),
        (3 / 16, 7.06),
        (3 / 8, 3.53),
    )

    def adjust(self, f_mul: float, w_mul: float) -> None:
        self.bands = tuple(map(lambda t: (t[0] * f_mul, t[1] * w_mul), self.bands))


class CHO:
    def __init__(
        self,
        channel_config: CHOChannelConfig | None = None,
        channel_freq_mul: float = 1,
        channel_width_mul: float = 1,
        internal_noise_std: float = 0,
        _debug_mode: bool = False,
    ) -> None:
        self.ch_cfg = channel_config
        self.ch_freq_mul = channel_freq_mul
        self.ch_width_mul = channel_width_mul

        self.noise_std = internal_noise_std

        self._debug_mode = _debug_mode

        if self.ch_cfg is None:
            self.ch_cfg = CHOChannelConfig()

        self.ch_cfg.adjust(self.ch_freq_mul, self.ch_width_mul)

        if self._debug_mode:
            makedirs(".temp/channels", exist_ok=True)
            makedirs(".temp/responces", exist_ok=True)

    def _build_channels(self, width: int, height: int) -> NDArray[np.float64]:
        thetas = [0, 45, 90, 135]
        channels = np.zeros((len(self.ch_cfg.bands) * len(thetas), height, width))

        for ci, ((f, w), theta) in enumerate(product(self.ch_cfg.bands, thetas)):

            def gabor_channel(i: int, j: int):
                return _gabor_filter(
                    i - height // 2, j - width // 2, f, w, np.deg2rad(theta)
                )

            channels[ci] = np.fromfunction(gabor_channel, (height, width))

            # Channel normalization
            channels[ci] /= np.sqrt(np.sum(channels[ci] * channels[ci]))

            if self._debug_mode:
                ch_img = channels[ci] - channels[ci].min()
                ch_img *= 255 / ch_img.max()
                ch_img = ch_img.astype(np.uint8)

                Image.fromarray(ch_img).save(
                    f".temp/channels/f{f:.2f}_w{w:.2f}_{theta}.png"
                )

        self.channels = channels

    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        self._build_channels(X.shape[2], X.shape[1])

        X_p: NDArray = X[y]
        X_n: NDArray = X[np.logical_not(y)]

        Nu_p = np.sum(self.channels[None, :, :, :] * X_p[:, None, :, :], axis=(2, 3))
        Nu_n = np.sum(self.channels[None, :, :, :] * X_n[:, None, :, :], axis=(2, 3))

        K = (np.cov(Nu_p, rowvar=False) + np.cov(Nu_n, rowvar=False)) / 2
        mean_nu = np.mean(Nu_p, axis=0) - np.mean(Nu_n, axis=0)

        self.template = np.linalg.inv(K) @ mean_nu

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        responses = np.sum(self.channels[None, :, :, :] * X[:, None, :, :], axis=(2, 3))
        t = np.sum(responses * self.template[np.newaxis, :], axis=1)

        if np.isclose(self.noise_std, 0):
            return t

        return t + np.random.normal(0, self.noise_std, t.shape)

    def measure(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> float:
        return float(metrics.roc_auc_score(y, self.test(X)))
