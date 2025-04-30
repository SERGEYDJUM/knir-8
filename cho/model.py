from copy import deepcopy
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
        channel_noise_std: float = 0,
        test_stat_noise_std: float = 0,
        train_set_keep: float = 1,
        _debug_mode: bool = False,
    ) -> None:
        self.ch_cfg = channel_config
        self.ch_noise_std = channel_noise_std
        self.test_noise_std = test_stat_noise_std
        self.tran_set_keep = train_set_keep
        self._debug_mode = _debug_mode

        if _debug_mode:
            makedirs(".temp/channels", exist_ok=True)

        if self.ch_cfg is None:
            self.ch_cfg = CHOChannelConfig()

        self.ch_cfg.adjust(channel_freq_mul, channel_width_mul)
        self.channels = None

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
            channels[ci] /= np.sqrt((channels[ci] * channels[ci]).sum())

            if self._debug_mode:
                ch_img = channels[ci] - channels[ci].min()
                ch_img *= 255 / ch_img.max()
                ch_img = ch_img.astype(np.uint8)

                Image.fromarray(ch_img).save(
                    f".temp/channels/f{f:.2f}_w{w:.2f}_{theta}.png"
                )

        return channels

    def _channelized_cov(self, X: NDArray) -> NDArray:
        # Get image variation
        x: NDArray = X - X.mean(axis=0)
        # Channelize variations
        x = np.sum(self.channels[None, :, :, :] * x[:, None, :, :], axis=(2, 3))
        # Return covariance matrix
        return (x.T @ x) / x.shape[0]

    def channel_responses(
        self,
        X: NDArray[np.float64],
        test: bool = False,
    ) -> NDArray[np.float64]:
        responses = np.sum(self.channels[None, :, :, :] * X[:, None, :, :], axis=(2, 3))

        if test:
            noise_std = self.Nu_n_std * self.ch_noise_std
            noise_std = np.broadcast_to(noise_std, responses.shape)
            responses += np.random.normal(0, noise_std, responses.shape)

        return responses

    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        self.channels = self._build_channels(X.shape[2], X.shape[1])

        X_p: NDArray = X[y]
        X_p = X_p[: int(X_p.shape[0] * self.tran_set_keep)]

        X_n: NDArray = X[np.logical_not(y)]
        X_n = X_n[: int(X_n.shape[0] * self.tran_set_keep)]

        mean_diff = np.mean(X_p, axis=0) - np.mean(X_n, axis=0)
        mean_diff_nu = self.channel_responses(mean_diff[np.newaxis, :, :])[0]

        self.Nu_n_std = np.std(self.channel_responses(X_n), axis=0)

        K_nu = (self._channelized_cov(X_p) + self._channelized_cov(X_n)) / 2
        self.template = np.linalg.inv(K_nu) @ mean_diff_nu

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.sum(
            self.channel_responses(X, test=True) * self.template[np.newaxis, :],
            axis=1,
        )

        return t + np.random.normal(0, self.test_noise_std, t.shape)

    def measure(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> float:
        t = self.test(X)
        return max(
            float(metrics.roc_auc_score(y, t)), float(metrics.roc_auc_score(y, -t))
        )


class CHOss(CHO):
    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        self.channels = self._build_channels(X.shape[2], X.shape[1])

        X_n: NDArray = X[np.logical_not(y)]
        X_n = X_n[: int(X_n.shape[0] * self.tran_set_keep)]

        Nu_n = self.channel_responses(X_n)
        K_nu_n = self._channelized_cov(X_n)

        self.mean_nu_n = np.mean(Nu_n, axis=0)
        self.Nu_n_std = np.std(Nu_n, axis=0)
        self.inv_cov_n = np.linalg.inv(K_nu_n)

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        responses = self.channel_responses(X, test=True) - self.mean_nu_n
        t = np.sum(responses.T * (self.inv_cov_n @ responses.T), axis=0)
        return t + np.random.normal(0, self.test_noise_std, t.shape)


class NPWMF(CHO):
    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        X_n: NDArray = X[np.logical_not(y)]
        X_n = X_n[: int(X_n.shape[0] * self.tran_set_keep)]

        X_p: NDArray = X[y]
        X_p = X_p[: int(X_p.shape[0] * self.tran_set_keep)]

        self.x_meandiff = X_p.mean(axis=0) - X_n.mean(axis=0)

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.sum(X * self.x_meandiff[np.newaxis, :, :], axis=(1, 2))
        return t + np.random.normal(0, self.test_noise_std, t.shape)


class CHOArray:
    def __init__(self, model: CHO) -> None:
        self.model_base = model
        self.models: list[CHO] = []

    def train(
        self, X: NDArray[np.single], y: NDArray[np.bool], disc: NDArray[np.uint8]
    ) -> None:

        N_places = np.max(disc) + 1
        self.models.clear()

        for i in range(N_places):
            place_mask = disc == i
            model = deepcopy(self.model_base)
            model.train(X[place_mask], y[place_mask])
            self.models.append(model)

    def test(
        self, X: NDArray[np.single], disc: NDArray[np.uint8]
    ) -> NDArray[np.single]:
        output = np.zeros(X.shape[0], dtype=np.single)

        for i, x_s, place_s in zip(range(X.shape[0]), X, disc):
            output[i] = self.models[place_s].test(x_s[np.newaxis, :, :])[0]

        return output

    def measure(
        self, X: NDArray[np.float64], y: NDArray[np.bool], disc: NDArray[np.uint8]
    ) -> float:
        return float(metrics.roc_auc_score(y, self.test(X, disc)))
