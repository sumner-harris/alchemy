import torch
from typing import Optional, Tuple, Dict
from botorch.acquisition import (
    qUpperConfidenceBound,
    qLogExpectedImprovement,
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qNegIntegratedPosteriorVariance,
)
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from .surrogates import RandomForestSurrogate, NeuroBayesBNNSurrogate, GPSurrogate


def step_surrogate(
    X: torch.Tensor,
    Y: torch.Tensor,
    X_test: torch.Tensor,
    bounds: Optional[torch.Tensor] = None,
    surrogate: Dict = None,
    acquisition: Optional[Dict] = None,
    return_acq_vals: bool = False,
    maximize: bool = True,
    **opt_args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit surrogate, compute posterior, and (optionally) optimize MC acquisition.

    surrogate dict keys:
      name: 'gp' | 'rf' | 'bnn'
      init_kwargs: passed to surrogate constructor
      fit_kwargs: passed to surrogate.fit

    acquisition dict keys:
      name: 'UCB' | 'EI' | 'LOGEI' | 'NOISY_EI' | 'REGRET' | 'UNC'
      kwargs: passed to acquisition constructor (e.g. sampler, beta)
      discrete: whether to use optimize_acqf_discrete

    Returns:
      candidates (q x d), mean (m x k), var (m x k), acq_vals (m) or empty
    """
    # 1) maximize/minimize
    Y_train = Y if maximize else -Y

    # 2) fit surrogate
    name = surrogate.get('name', 'gp').lower()
    init_kw = surrogate.get('init_kwargs', {})
    fit_kw = surrogate.get('fit_kwargs', {})
    if name == 'gp':
        model = GPSurrogate(**init_kw).fit(X, Y_train, **fit_kw)
    elif name == 'rf':
        model = RandomForestSurrogate(**init_kw).fit(X, Y_train, **fit_kw)
    elif name == 'bnn':
        model = NeuroBayesBNNSurrogate(**init_kw).fit(X, Y_train, **fit_kw)
    else:
        raise ValueError(f"Unknown surrogate: {name}")

    # 3) posterior on X_test
    posterior = model.posterior(X_test)
    mean = posterior.mean.detach()
    var  = posterior.variance.detach()
    if not maximize:
        mean = -mean

    # 3.5) Ensure acquisition uses standardized Y for non-GP models
    if hasattr(model, 'outcome_transform') and name in ('rf', 'bnn'):
        Y_train = model.outcome_transform(Y_train)[0]

    candidates = None
    acq_vals = torch.tensor([])

    # 4) optional acquisition
    if acquisition:
        mode     = acquisition.get('name', 'UCB').strip().upper()
        acq_kw   = acquisition.get('kwargs', {}).copy()
        discrete = acquisition.get('discrete', False)
        q        = acq_kw.pop('q', opt_args.get('batch', 1))

        # build acquisition function
        if mode == 'UCB':
            acqf = qUpperConfidenceBound(model, **acq_kw)
        elif mode in ('EI', 'EXPECTED_IMPROVEMENT', 'LOGEI', 'QLOGEI'):
            best_f = Y_train.max().item()
            acqf   = qLogExpectedImprovement(model, best_f, **acq_kw)
        elif mode in ('NOISY_EI', 'QNEI'):
            acqf   = qNoisyExpectedImprovement(model, X_baseline=X, **acq_kw)
        elif mode == 'REGRET':
            acqf   = qSimpleRegret(model, **acq_kw)
        elif mode in ('UNC', 'MAXVAR', 'INTEGRATED_VAR'):
            # We'll use posterior variance for uncertainty acquisition (enumeration)
            acqf = None
        else:
            raise ValueError(f"Unsupported acquisition: {mode}")

        # attach pending points for fantasy samplers if needed
        if acqf is not None:
            acqf.set_X_pending(X)

        # optimize or enumerate based on surrogate type and mode
        if discrete:
            candidates, acq_vals = optimize_acqf_discrete(
                acq_function=acqf,
                choices=X_test,
                q=q,
            )
        elif mode in ('UNC', 'MAXVAR', 'INTEGRATED_VAR'):
            # uncertainty acquisition via posterior variance
            acq_vals = var.mean(dim=-1)
            topk = torch.topk(acq_vals, k=q)
            candidates = X_test[topk.indices]
        elif name in ('rf', 'bnn'):
            # non-differentiable surrogates: enumeration via acquisition function
            acq_vals = evaluate_in_batches(
                acqf,
                X_test,
                batch_size=opt_args.get('predict_batch_size', 100)
            )
            topk = torch.topk(acq_vals, k=q)
            candidates = X_test[topk.indices]
        else:
            # continuous GP optimization for other modes
            if bounds is None:
                raise ValueError("`bounds` required for continuous acquisition")
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=opt_args.get('num_restarts', 5),
                raw_samples=opt_args.get('raw_samples', 20),
            )
            if return_acq_vals and acqf is not None:
                acq_vals = evaluate_in_batches(
                    acqf,
                    X_test,
                    batch_size=opt_args.get('predict_batch_size', 100)
                )

    return candidates, mean, var, acq_vals


def evaluate_in_batches(acqf, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    """Evaluate acquisition in batches to avoid OOM."""
    vals = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            batch = X[i : i + batch_size]
            acq_val = acqf(batch.unsqueeze(-2)).squeeze(-1)
            vals.append(acq_val)
    return torch.cat(vals)
