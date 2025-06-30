import torch
from typing import Optional, Tuple, Dict
from botorch.acquisition import (
    qUpperConfidenceBound,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qSimpleRegret,
    qNegIntegratedPosteriorVariance,
)
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_discrete_local_search
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
    Fit a surrogate model to training data, construct an acquisition function,
    and propose new candidate points by optimizing or scoring the acquisition.

    Parameters
    ----------
    X : torch.Tensor
        Training inputs (N x d).
    Y : torch.Tensor
        Training targets (N x 1).
    X_test : torch.Tensor
        Candidate/test points (M x d) for discrete evaluation.
    bounds : Optional[torch.Tensor]
        Bounds for each input dimension (2 x d), used for continuous optimization.
    surrogate : Dict
        Dictionary specifying the surrogate model.
            - name: 'gp', 'rf', or 'bnn'
            - init_kwargs: constructor kwargs
            - fit_kwargs: fitting kwargs
    acquisition : Optional[Dict]
        Dictionary specifying the acquisition function and optimizer.
            - name: 'EI', 'UCB', 'LOGEI', 'NEI', 'UNC', etc.
            - kwargs: acquisition constructor kwargs (e.g. q, beta)
            - discrete: bool, whether to use discrete search
            - optimizer_kwargs: kwargs for the acquisition optimizer
    return_acq_vals : bool
        Whether to return acquisition values evaluated on X_test.
    maximize : bool
        Whether the objective is to be maximized or minimized.
    **opt_args : Any
        Extra args passed to acquisition optimizer if needed.

    Returns
    -------
    candidates : torch.Tensor
        Selected next query points (q x d).
    mean : torch.Tensor
        Posterior mean predictions at X_test.
    var : torch.Tensor
        Posterior variance predictions at X_test.
    acq_vals : torch.Tensor
        Acquisition values evaluated on X_test (if requested).
    """
    # -----------------------------------
    # Preprocess and fit surrogate model
    # -----------------------------------
    Y_train = Y if maximize else -Y
    surrogate = surrogate or {}
    acquisition = acquisition or {}

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

    posterior = model.posterior(X_test)
    mean = posterior.mean.detach()
    var = posterior.variance.detach()
    if not maximize:
        mean = -mean

    candidates = None
    acq_vals = torch.tensor([])

    # -------------------------------
    # Build acquisition function
    # -------------------------------
    if acquisition:
        acq_kw   = acquisition.get('kwargs', {}).copy()
        opt_args = acquisition.get('optimizer_kwargs', {})
        mode     = acquisition.get('name', 'EI').strip().upper()
        discrete = acquisition.get('discrete', False)
        q        = acq_kw.pop('q', opt_args.get('batch', 1))

        # Force discrete mode for non-differentiable models
        if name in ('rf', 'bnn') and not discrete:
            print(f"Warning: surrogate '{name}' is non-differentiable. Forcing discrete=True.")
            discrete = True

        # Select acquisition function
        if mode == 'UCB':
            acq_kw.setdefault('beta', 0.2)
            acqf = qUpperConfidenceBound(model, **acq_kw)
        elif mode in ('EI', 'LOGEI', 'QLOGEI'):
            best_f = Y_train.max().item()
            acqf = qLogExpectedImprovement(model, best_f, **acq_kw)
        elif mode in ('NEI', 'LOGNEI'):
            acqf = qLogNoisyExpectedImprovement(model, X_baseline=X, **acq_kw)
        elif mode == 'REGRET':
            acqf = qSimpleRegret(model, **acq_kw)
        elif mode in ('UNC', 'MAXVAR'):
            print(f"Note: '{mode}' uses posterior variance, not an acquisition function.")
            acqf = None
        else:
            raise ValueError(f"Unsupported acquisition: {mode}")

        # Register pending points
        if acqf is not None:
            acqf.set_X_pending(X)

        # -------------------------------
        # Optimize or enumerate acquisition
        # -------------------------------
        if discrete:
            # Discrete search: use X_test or local search
            if mode in ('UNC', 'MAXVAR'):
                acq_vals = var.mean(dim=-1)
                topk = torch.topk(acq_vals, k=q)
                candidates = X_test[topk.indices]
            elif bounds is not None:
                d = bounds.shape[1]
                n_points = opt_args.get("n_points_per_dim", 20)
                discrete_choices = [
                    torch.linspace(bounds[0, i].item(), bounds[1, i].item(), n_points)
                    for i in range(d)
                ]
                num_restarts = opt_args.get("num_restarts", 10)
                raw_samples  = opt_args.get("raw_samples", 4096 * d)

                print('Running local search Acqf Opt')
                candidates, acq_vals = optimize_acqf_discrete_local_search(
                    acq_function=acqf,
                    discrete_choices=discrete_choices,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            else:
                print('Running grid search Acqf Opt')
                candidates, acq_vals = optimize_acqf_discrete(
                    acq_function=acqf,
                    choices=X_test,
                    q=q,
                )
        else:
            # Continuous optimization
            if mode in ('UNC', 'MAXVAR'):
                acq_vals = var.mean(dim=-1)
                topk = torch.topk(acq_vals, k=q)
                candidates = X_test[topk.indices]
            else:
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
                        batch_size=opt_args.get('predict_batch_size', 1000)
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