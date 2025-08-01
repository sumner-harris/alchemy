import pandas as pd
import torch
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

def read_print_data(file_path: str,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads the Excel at `file_path` (and optional `sheet_name`), 
    extracts the columns:
       X = [Speed, Sintering Power (const), Printing Passes, Sintering Passes]
       Y = Average Resistance
       Y_var = Std Dev
    Converts them into torch tensors of shape (N,4), (N,1), (N,1),
    prints their shapes, and returns (X, Y, Y_var).
    """
    # 1) Load into a DataFrame
    df = pd.read_excel(file_path)
    
    # 2) Define which columns go where
    X_cols    = ['Speed',
                 'Sintering Power (const)',
                 'Printing Passes',
                 'Sintering Passes']
    Y_col     = 'Average Resistance'
    Yvar_col  = 'Std Dev'
    
    # 3) Extract numpy arrays
    X_np    = df[X_cols].to_numpy(dtype=float)     # (N,4)
    Y_np    = df[[Y_col]].to_numpy(dtype=float)    # (N,1)
    Yvar_np = df[[Yvar_col]].to_numpy(dtype=float) # (N,1)
    
    # 4) Convert to torch tensors
    X     = torch.from_numpy(X_np).double()     # torch.Size([N,4])
    Y     = torch.from_numpy(Y_np).double()     # torch.Size([N,1])
    Y_var = torch.from_numpy(Yvar_np).double()  # torch.Size([N,1])
    
    # 5) Print to verify
    print(f"X: {X.shape}, Y: {Y.shape}, Y_var: {Y_var.shape}")
    
    return X, Y, Y_var

def plot_surrogate(
    mean: torch.Tensor,
    grids: list,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    var: torch.Tensor = None,
    param_names: list[str] = None,
    figsize: tuple[float, float] = (12, 12),
):
    """
    Upper‐triangle pairwise GP‐mean projections in a (D–1)x(D–1) GridSpec.
    • Each axes box is square (set_box_aspect(1)).
    • Data uses aspect='auto' so wildly different scales still plot correctly.
    • All offset/scientific tick notation is disabled.
    • EVERY subplot shows its x- and y-label.
    """
    D = len(grids)
    if param_names is None:
        param_names = [f"x{i}" for i in range(D)]
        
    # ── 1D special case ─────────────────────────────────────────────────────────
    if D == 1:
        # extract grid and mean
        gi    = grids[0].detach().cpu().numpy() \
                 if torch.is_tensor(grids[0]) else np.asarray(grids[0])
        mean1 = mean.detach().cpu().numpy().reshape(-1)

        # if var passed, turn into std‐array
        if var is not None:
            var1 = var.detach().cpu().numpy().reshape(-1)
            std1 = np.sqrt(var1)
        else:
            std1 = None

        # plot
        fig, ax = plt.subplots(figsize=figsize)

        # fill‐between for 95% CI if we have variance
        if std1 is not None:
            lower = mean1 - 1.96 * std1
            upper = mean1 + 1.96 * std1
            ax.fill_between(gi, lower, upper, alpha=0.3, label="95% CI")

        # mean line + training points
        ax.plot(gi, mean1, "-", label="GP mean")
        ax.scatter(
            X_train[:, 0].cpu().numpy(),
            Y_train.squeeze(-1).cpu().numpy(),
            c=Y_train.squeeze(-1).cpu().numpy(),
            cmap="viridis",
            edgecolor="k",
            label="observations",
        )

        ax.set_xlabel(param_names[0])
        ax.set_ylabel("GP mean")
        ax.legend(loc="best")
        plt.show()
        return
    # ────────────────────────────────────────────────────────────────────────────

    # reshape mean → (n0, n1, ..., n_{D-1})
    shapes  = [int(g.shape[0]) for g in grids]
    mean_np = mean.detach().cpu().numpy().reshape(*shapes)
    # reshape var → D-dim array if given
    var_np  = var.detach().cpu().numpy().reshape(*shapes) if var is not None else None

    # ensure numpy
    grid_np = [
        g.detach().cpu().numpy() if torch.is_tensor(g) else np.asarray(g)
        for g in grids
    ]

    # grid for upper triangle has (D-1) rows and (D-1) cols
    n_rows = n_cols = D - 1

    # recompute figsize so each cell ends up square
    total_w, total_h = figsize
    cell  = min(total_w / n_cols, total_h / n_rows)
    figsize_new = (cell * n_cols, cell * n_rows)

    fig = plt.figure(figsize=figsize_new)
    gs  = GridSpec(
        n_rows, n_cols, figure=fig,
        width_ratios =[1]*n_cols,
        height_ratios=[1]*n_rows,
        wspace=0.6, hspace=0.2
    )

    for i in range(D-1):
        for j in range(i+1, D):
            row = i
            col = j - i - 1
            ax  = fig.add_subplot(gs[row, col])

            # average out all dims except i,j
            other = [k for k in range(D) if k not in (i, j)]
            proj  = mean_np.mean(axis=tuple(other))  # shape = (len(grid_i), len(grid_j))

            gi, gj = grid_np[i], grid_np[j]

            # draw the heatmap
            im = ax.imshow(
                proj.T,
                origin='lower',
                aspect='auto',               # data units scale freely
                extent=(gi[0], gi[-1], gj[0], gj[-1]),
            )

            # scatter training points
            ax.scatter(
                X_train[:, i].cpu().numpy(),
                X_train[:, j].cpu().numpy(),
                c=Y_train.squeeze(-1).cpu().numpy(),
                cmap='viridis',
                edgecolor='k',
            )

            # 1) square **box**, but keep data aspect auto
            ax.set_box_aspect(1)

            # 2) zero margins so heatmap fills the box
            ax.margins(0)
            ax.set_xlim(gi[0], gi[-1])
            ax.set_ylim(gj[0], gj[-1])

            # 3) disable any offset/scientific notation
            ax.ticklabel_format(style="plain", useOffset=False)
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

            # **always** label both axes
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])

            ax.set_title(f"{param_names[i]} vs {param_names[j]}", pad=2, fontsize="small")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()
    
    # ── if we have var, do it all again for the variance ───────────────────────
    if var_np is not None:
        fig = plt.figure(figsize=figsize_new)
        gs  = GridSpec(
            n_rows, n_cols, figure=fig,
            width_ratios =[1]*n_cols,
            height_ratios=[1]*n_rows,
            wspace=0.6, hspace=0.2
        )
        for i in range(D-1):
            for j in range(i+1, D):
                row, col = i, j - i - 1
                ax = fig.add_subplot(gs[row, col])

                # compute the variance projection
                other    = [k for k in range(D) if k not in (i, j)]
                proj_var = var_np.mean(axis=tuple(other))

                gi, gj = grid_np[i], grid_np[j]
                im = ax.imshow(
                    proj_var.T,
                    origin='lower',
                    aspect='auto',
                    extent=(gi[0], gi[-1], gj[0], gj[-1]),
                )
                # same scatter of training points
                ax.scatter(
                    X_train[:, i].cpu().numpy(),
                    X_train[:, j].cpu().numpy(),
                    c=Y_train.squeeze(-1).cpu().numpy(),
                    cmap='viridis',
                    edgecolor='k',
                )

                ax.set_box_aspect(1)
                ax.margins(0)
                ax.set_xlim(gi[0], gi[-1])
                ax.set_ylim(gj[0], gj[-1])

                # disable scientific notation
                ax.ticklabel_format(style="plain", useOffset=False)
                fmt = ScalarFormatter(useOffset=False)
                fmt.set_scientific(False)
                ax.xaxis.set_major_formatter(fmt)
                ax.yaxis.set_major_formatter(fmt)

                ax.set_xlabel(param_names[i])
                ax.set_ylabel(param_names[j])
                ax.set_title(f"Var: {param_names[i]} vs {param_names[j]}", pad=2, fontsize="small")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()
    
def plot_gp_mean_projections(
    mean: torch.Tensor,
    grids: list,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    param_names: list[str] = None,
    figsize=(15, 5),
):
    """
    Plot all pairwise 2D projections of a GP posterior mean for N-dimensional inputs.

    Args:
        mean:       torch.Tensor of shape (prod(n_i),) containing the posterior mean 
                    values on the full grid (flattened).
        grids:      list of N one-dimensional torch.Tensor or array, each of length n_i.
        X_train:    torch.Tensor of shape (M, N) with the training inputs.
        Y_train:    torch.Tensor of shape (M, 1) or (M,) with the training outputs.
        param_names: optional list of N strings for axis labels; defaults to ["x0", "x1", …].
        figsize:    size of the full figure (width, height).

    Notes:
        • The function arranges subplots in 3 columns and as many rows as needed.
        • Each subplot shows the posterior mean averaged over the other N−2 dimensions.
    """
    D = len(grids)
    # default names
    if param_names is None:
        param_names = [f"x{i}" for i in range(D)]

    # reshape the flat mean into an N-dim array
    shapes = [int(g.shape[0]) for g in grids]
    mean_np = mean.detach().cpu().numpy().reshape(*shapes)

    # prepare numpy versions of the grids
    grid_np = [
        g.detach().cpu().numpy() if torch.is_tensor(g) else np.asarray(g)
        for g in grids
    ]

    # all combinations of two distinct dims
    combos = list(itertools.combinations(range(D), 2))
    n_plots = len(combos)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (i, j) in enumerate(combos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # average out all other axes
        other_axes = [k for k in range(D) if k not in (i, j)]
        proj = mean_np.mean(axis=tuple(other_axes))  # shape (n_i, n_j)

        # create mesh for dims i,j
        Gi, Gj = np.meshgrid(grid_np[i], grid_np[j], indexing="ij")

        pcm = ax.pcolormesh(Gi, Gj, proj, shading="auto")
        ax.scatter(
            X_train[:, i].cpu().numpy(),
            X_train[:, j].cpu().numpy(),
            c=Y_train.squeeze(-1).cpu().numpy(),
            edgecolor="k",
            cmap="viridis",
        )
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_title(f"{param_names[i]} vs {param_names[j]}")
        fig.colorbar(pcm, ax=ax)

    # hide any unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.show()
