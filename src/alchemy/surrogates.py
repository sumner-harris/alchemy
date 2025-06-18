# src/alchemy/surrogates.py
from typing import Optional, Tuple, Dict
import torch
import numpy as np
from botorch.models.model import Model
from botorch.posteriors.torch import TorchPosterior
from gpytorch.settings import cholesky_jitter
from torch.distributions import Normal
from sklearn.ensemble import RandomForestRegressor
import neurobayes as nb
from .gp_models import create_single_task_gp
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import IIDNormalSampler


class EnsemblePosterior(TorchPosterior):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        distribution = Normal(mean, std)
        super().__init__(distribution)
        self.mean     = mean
        self.stddev   = std
        self.variance = std.pow(2)

    @property
    def batch_shape(self) -> torch.Size:
        return self.mean.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([self.mean.shape[-1]])

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.batch_shape + self.event_shape

    @property
    def batch_range(self) -> Tuple[int,int]:
        return (0, 0)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.distribution.rsample(sample_shape)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: torch.Tensor
    ) -> torch.Tensor:
        return base_samples * self.stddev + self.mean


# register IIDNormalSampler for any Normal‐based TorchPosterior
@GetSampler.register(Normal)
def _get_sampler_normal(posterior, sample_shape: torch.Size, seed: int = None):
    return IIDNormalSampler(sample_shape=sample_shape, seed=seed)


class RandomForestSurrogate(Model):
    """A bootstrap‐ensemble RF that exposes an MC‐capable posterior."""
    def __init__(self, n_estimators: int = 10, **rf_kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.rf_kwargs = rf_kwargs
        self.models = []
        self._num_outputs = None

    @property
    def num_outputs(self) -> int:
        if self._num_outputs is None:
            raise RuntimeError("RandomForestSurrogate must be fit before querying num_outputs")
        return self._num_outputs

    def fit(self, X: torch.Tensor, Y: torch.Tensor, **fit_kwargs) -> "RandomForestSurrogate":
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        if Y_np.ndim == 1:
            Y_np = Y_np.reshape(-1, 1)
        n, k = Y_np.shape
        self._num_outputs = k
        self.models = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n, n, replace=True)
            rf = RandomForestRegressor(**self.rf_kwargs)
            rf.fit(X_np[idxs], Y_np[idxs].ravel())
            self.models.append(rf)
        return self

    def posterior(self, X: torch.Tensor, observation_noise: bool = False, **kwargs):
        batch_dims = X.shape[:-1]
        d = X.shape[-1]
        X_flat = X.reshape(-1, d).detach().cpu().numpy()
        preds = np.stack([m.predict(X_flat) for m in self.models], axis=0)
        mean_np = preds.mean(axis=0)
        std_np = preds.std(axis=0) + 1e-6
        mean = torch.from_numpy(mean_np).float().reshape(*batch_dims, -1)
        std = torch.from_numpy(std_np).float().reshape(*batch_dims, -1)
        return EnsemblePosterior(mean, std)


class NeuroBayesBNNSurrogate(Model):
    """A NeuroBayes BNN with MC-capable posterior."""
    def __init__(self, architecture, num_warmup: int = 500, num_samples: int = 500, **fit_kwargs):
        super().__init__()
        self.architecture = architecture
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.fit_kwargs = fit_kwargs
        self.bnn_model = None
        self._num_outputs = None

    @property
    def num_outputs(self) -> int:
        if self._num_outputs is None:
            raise RuntimeError("NeuroBayesBNNSurrogate must be fit before querying num_outputs")
        return self._num_outputs

    def fit(self, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> "NeuroBayesBNNSurrogate":
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        if Y_np.ndim == 1:
            Y_np = Y_np.reshape(-1, 1)
        n, k = Y_np.shape
        self._num_outputs = k
        bnn = nb.BNN(self.architecture)
        bnn.fit(X_np, Y_np, num_warmup=self.num_warmup, num_samples=self.num_samples, **self.fit_kwargs)
        self.bnn_model = bnn
        return self

    def posterior(self, X: torch.Tensor, observation_noise: bool = False, **kwargs):
        batch_dims = X.shape[:-1]
        d = X.shape[-1]
        X_flat = X.reshape(-1, d).detach().cpu().numpy()
        mean_np, var_np = self.bnn_model.predict(X_flat)
        mean_np = np.array(mean_np)
        std_np  = np.sqrt(np.array(var_np)) + 1e-6
        mean = torch.from_numpy(mean_np).float().reshape(*batch_dims, -1)
        std = torch.from_numpy(std_np).float().reshape(*batch_dims, -1)
        return EnsemblePosterior(mean, std)


class GPSurrogate(Model):
    """Thin wrapper around SingleTaskGP (only single-output)."""
    def __init__(self, ard: bool = True):
        super().__init__()
        self.ard = ard
        self.gp = None

    @property
    def num_outputs(self) -> int:
        if self.gp is None:
            raise RuntimeError("GPSurrogate must be fit before querying num_outputs")
        return 1

    def fit(self, X: torch.Tensor, Y: torch.Tensor, Y_var: torch.Tensor = None) -> "GPSurrogate":
        if Y.dim() == 2 and Y.size(1) != 1:
            raise ValueError(f"GPSurrogate only supports single-output. Got Y shape {tuple(Y.shape)}")
        with cholesky_jitter(1e-9):
            gp = create_single_task_gp(train_X=X, train_Y=Y, train_Yvar=Y_var, ard=self.ard)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        self.gp = gp
        return self

    def posterior(self, X: torch.Tensor, **kwargs):
        if self.gp is None:
            raise RuntimeError("GPSurrogate must be fit before calling posterior")
        return self.gp.posterior(X)
