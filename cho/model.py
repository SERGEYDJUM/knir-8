from copy import deepcopy
from itertools import product
from dataclasses import dataclass
from numpy.typing import NDArray
from sklearn import metrics
import numpy as np


def _gabor_filter(x: int, y: int, f: float, w: float, theta: float) -> float:
    """Calculates Gabor filter response at some point"""
    exp_arg = -4 * np.log(2) * (x * x + y * y) / (w * w)
    cos_arg = 2 * np.pi * f * (x * np.cos(theta) + y * np.sin(theta))
    return np.exp(exp_arg) * np.cos(cos_arg)


@dataclass
class CHOChannelConfig:
    """Gabor channel configuration for CHO-derived models"""

    CH_THETAS = [0, 45, 90, 135]

    channels: tuple[tuple[float, float]] = (
        (3 / 128, 56.48),
        (3 / 64, 28.24),
        (3 / 32, 14.12),
        (3 / 16, 7.06),
        (3 / 8, 3.53),
    )


class CHO:
    """Channelized Hotelling Observer model with Gabor channels"""

    def __init__(
        self,
        channel_config: CHOChannelConfig | None = None,
        channel_noise_mul: float = 0,
        teststat_noise_std: float = 0,
    ) -> None:
        self.ch_cfg = channel_config
        self.ch_noise_mul = channel_noise_mul
        self.test_noise_std = teststat_noise_std
        self.channels: NDArray = None
        self.channel_deviations: float = None

        if self.ch_cfg is None:
            self.ch_cfg = CHOChannelConfig()

    def _build_channels(self, width: int, height: int) -> None:
        """Generates channel template images as tensor"""
        channel_cfgs = list(product(self.ch_cfg.channels, self.ch_cfg.CH_THETAS))
        channels = np.zeros((len(channel_cfgs), height, width))

        for ci, ((f, w), theta) in enumerate(channel_cfgs):
            channels[ci] = np.fromfunction(
                lambda i, j: _gabor_filter(
                    i - height // 2, j - width // 2, f, w, np.deg2rad(theta)
                ),
                (height, width),
            )
            channels[ci] /= np.sqrt((channels[ci] * channels[ci]).sum())
        self.channels = channels

    def _channelized_cov(self, X: NDArray) -> NDArray:
        """Correctly calculates covariance matrix for images"""
        x: NDArray = X - X.mean(axis=0)
        x = np.sum(self.channels[None, :, :, :] * x[:, None, :, :], axis=(2, 3))
        return (x.T @ x) / x.shape[0]

    def _channel_responses(
        self, X: NDArray[np.float64], add_noise: bool = False
    ) -> NDArray[np.float64]:
        """Calculates channel responses with optional noise"""
        responses = np.sum(self.channels[None, :, :, :] * X[:, None, :, :], axis=(2, 3))
        if add_noise:
            noise_std = self.channel_deviations * self.ch_noise_mul
            noise_std = np.broadcast_to(noise_std, responses.shape)
            responses += np.random.normal(0, noise_std, responses.shape)
        return responses

    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        """Train a model. Will use all supplied information"""
        self._build_channels(X.shape[2], X.shape[1])

        X_p: NDArray = X[y]
        X_n: NDArray = X[np.logical_not(y)]

        mean_diff = np.mean(X_p, axis=0) - np.mean(X_n, axis=0)
        mean_diff_nu = self._channel_responses(mean_diff[np.newaxis, :, :])[0]
        K_nu = (self._channelized_cov(X_p) + self._channelized_cov(X_n)) / 2

        self.channel_deviations = np.std(self._channel_responses(X_n), axis=0)
        self.template = np.linalg.inv(K_nu) @ mean_diff_nu

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Caclulate test statistics for supplied images using model"""
        resp = self._channel_responses(X, add_noise=True)
        t = np.sum(resp * self.template[np.newaxis, :], axis=1)
        return t + np.random.normal(0, self.test_noise_std, t.shape)

    def measure(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> float:
        """Calculate ROC AUC for supplied images using model"""
        return float(metrics.roc_auc_score(y, self.test(X)))


class CHOss(CHO):
    """Single Sample Channelized Hotelling Observer model with Gabor channels"""

    def train(self, X: NDArray[np.float64], y: NDArray[np.bool]) -> None:
        """Train a model. Will use only images without signal"""
        self._build_channels(X.shape[2], X.shape[1])
        X_n: NDArray = X[np.logical_not(y)]
        Nu_n = self._channel_responses(X_n)
        K_nu_n = self._channelized_cov(X_n)

        self.mean_nu_n = np.mean(Nu_n, axis=0)
        self.channel_deviations = np.std(Nu_n, axis=0)
        self.inv_cov_n = np.linalg.inv(K_nu_n)

    def test(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Caclulate test statistics for supplied images using model"""
        responses = self._channel_responses(X, add_noise=True) - self.mean_nu_n
        t = np.sum(responses.T * (self.inv_cov_n @ responses.T), axis=0)
        return t + np.random.normal(0, self.test_noise_std, t.shape)


class CHOArray:
    """Class that allows specifying a Model Observers for each signal location"""

    def __init__(self, model: CHO | CHOss) -> None:
        self.model_base = model
        self.models: list[CHO] = []

    def train(
        self, X: NDArray[np.single], y: NDArray[np.bool], location: NDArray[np.uint8]
    ) -> None:

        n_locations = np.max(location) + 1
        self.models.clear()

        for i in range(n_locations):
            location_mask = location == i
            model = deepcopy(self.model_base)
            model.train(X[location_mask], y[location_mask])
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
