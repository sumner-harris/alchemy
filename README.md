# ALchemy
ALchemy /ˈæl.kə.mi/ (noun): a process so effective that it seems like magic.

ALchemy is a package for active learning built with experimentalists in physics, chemistry, and materials science in mind.
The goal is to provide one interface for calculating various surrogate models (from various packages) and acquisition functions with as little 
code as possible to lower the barrier to adopt modern AI-driven experimentation practices.

# Install Instructions
I reccomend using uv to create a new virtual environment and manage dependencies. First install uv https://docs.astral.sh/uv/getting-started/installation/ 

You can then install ALchemy by running:
```bash
uv pip install git+https://github.com/sumner-harris/alchemy.git
```

## Example

Here’s how you’d define the core “step_surrogate” function:

```python
from alchemy import step_surrogate
import itertools
import torch

# Assuming that you have a torch tensors X_train of shape (num_samples,num_dimensions),
# Y_train of shape (num_samples, 1), and an X_test of shape(num_points_to_eval, num_dimensions).
X_train = torch.rand(100,4, dtype=torch.float64)
Y_train = torch.rand(100,1, dtype=torch.float64)

bounds = torch.tensor([    # must provide bounds for acquisition function
    [0,  300,  1,  0.5],   # lower bounds for each parameter
    [50, 700, 2.5, 1.5],   # upper bounds: same order as above
], dtype=torch.float64)

# Create spaced points per dimension if you want a full grid search
# or sample points from the space in a better way.
num_points_in_grid = 20
grid_points = [torch.linspace(bounds[0, i], bounds[1, i], num_points_in_grid) for i in range(bounds.size(1))]

# Create Cartesian product of all grid points
X_test = torch.tensor(list(itertools.product(*grid_points)), dtype=torch.float64)

# choose your surrogate model, here a Gaussian process
surrogate = {
    'name': 'gp'
}

#Choose your acquisition function, batch size, and parameters
acquisition = {
    'name': 'UCB',
    'kwargs': {
        'q': 4, #input your batch size here, in this example, a batch of 4 
        'beta': 2,# set beta for UCB
    }
}

candidates, surrogate_mean, surrogate_variance, acquisition_vals = step_surrogate(
    X=X_train,
    Y=Y_train,
    X_test=X_test,
    bounds=bounds,
    surrogate=surrogate,
    acquisition=acquisition,
    maximize=True, # or False for a minimization problem
)
```

Currently also support sci-kit random forests with bootstrap aggregation for uncertainty.

```python
surrogate = {
    'name': 'rf',
    'init_kwargs': {
        'n_bootstrap': 10, # this is the number of individual RFs to average
        'n_trees': 100, # this the number of trees per RF
        # any other sklearn RF kwargs, e.g. 'max_depth': 5
    }
}
```
And also support NeuroBayes (Jax-based) fully Bayesian Neural Networks

```python
import neurobayes as nb;

# Initialize NN architecture
architecture = nb.FlaxMLP(hidden_dims = [8,8], target_dim=1)

surrogate = {
    'name': 'bnn',
    'init_kwargs': {
        'architecture': architecture,
        'num_warmup': 500,
        'num_samples': 500,
        # any other fit_kwargs for NeuroBayes BNN
    }
}
```
