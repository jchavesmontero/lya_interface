"""Figures of a Cobaya graph at a point (P1D, Vega, star-parameter ellipses).

Importing this module sets the house matplotlib style (STIX fonts).
"""

import numpy as np
from matplotlib import rcParams
from lya_interface.errors import propagate_local_errors
from lya_interface.fit_core import (cup1d_component, evaluate_point, format_goodness_of_fit,
                                    goodness_of_fit, is_blinded, star_values, vega_component,
                                    vega_inputs)

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


def plot_point(model, point, *, residuals=False, title=None):
    """Plot every selected redshift with cup1d's maintained panel renderers.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph with one cup1d likelihood.
    point : mapping
        Physical sampled values.
    residuals : bool, default=False
        Plot residual panels rather than power spectra.
    title : str, optional
        Prefix for the correlated joint chi-squared title.

    Returns
    -------
    tuple
        Matplotlib figure and axes.

    Error bars and per-bin chi2 are marginal block diagnostics; their sum need
    not equal the correlated joint chi2 displayed in the figure title.
    Neither fitted cosmology nor unblinded derived parameters are displayed.
    """
    from cup1d.likelihood.external import gaussian_residual
    from cup1d.postprocessing.p1d import P1DBin, plot_p1d_spectra, plot_p1d_residuals

    result, powers = evaluate_point(model, point)
    backend = cup1d_component(model).backend
    bins = []
    for key, data in backend.data.items():
        for i, z in enumerate(data.z):
            chi2, _ = gaussian_residual(
                np.asarray(data.Pk_kms[i]) - powers[key][i], backend.chol_Pk_kms[key][i]
            )
            bins.append(
                P1DBin(
                    key,
                    i,
                    float(z),
                    np.asarray(data.k_kms[i]),
                    np.asarray(data.Pk_kms[i]),
                    np.asarray(powers[key][i]),
                    np.sqrt(np.diag(backend.cov_Pk_kms[key][i])),
                    chi2,
                    len(data.k_kms[i]),
                    "C0",
                    0.0,
                )
            )
    renderer = plot_p1d_residuals if residuals else plot_p1d_spectra
    fig, axes = renderer(bins, panels=True, fontsize=14, print_chi2=False)
    statistics = goodness_of_fit(model, point)
    prefix = "" if title is None else title + "\n"
    fig.suptitle(
        prefix + format_goodness_of_fit(statistics, multiline=True)
    )
    fig.supxlabel(r"$k_\parallel$ [s/km]")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    return fig, axes


def plot_point_with_residuals(model, point, ncols=4, title=None):
    """Plot ``k P1D / pi`` and the relative residual for every redshift.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph with one cup1d likelihood.
    point : mapping
        Physical sampled values.
    ncols : int, default=4
        Number of redshift columns. Each redshift has a spectrum panel above a
        residual panel sharing the ``k`` axis.
    title : str, optional
        Prefix for the correlated joint chi-squared title.

    Returns
    -------
    tuple
        Matplotlib figure and the array of spectrum axes, one per redshift.

    Notes
    -----
    The model includes contaminants, resolution and rebinning onto the data
    bins. Panels are grouped by data set (named in their titles when there are
    several), then ordered in redshift. Residuals are ``data / model - 1`` with
    marginal (diagonal) errors
    ``sigma / model``; the per-redshift chi2 uses the same marginal blocks, so
    their sum need not equal the correlated joint chi2 in the title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from cup1d.likelihood.external import gaussian_residual

    result, powers = evaluate_point(model, point)
    backend = cup1d_component(model).backend
    entries = []
    several = len(backend.data) > 1
    for order, (key, data) in enumerate(backend.data.items()):
        for i, z in enumerate(data.z):
            k = np.asarray(data.k_kms[i])
            pk = np.asarray(data.Pk_kms[i])
            mod = np.asarray(powers[key][i])
            err = np.sqrt(np.diag(backend.cov_Pk_kms[key][i]))
            chi2, _ = gaussian_residual(pk - mod, backend.chol_Pk_kms[key][i])
            label = f"{key} " if several else ""
            entries.append((order, float(z), label, k, pk, mod, err, chi2))
    # grouped by data set, then by redshift
    entries.sort(key=lambda e: (e[0], e[1]))
    entries = [entry[1:] for entry in entries]
    nrows = -(-len(entries) // ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, left=0.06, right=0.99, bottom=0.05,
                             top=0.88, wspace=0.28, hspace=0.3)
    spectra = []
    for n, (z, label, k, pk, mod, err, chi2) in enumerate(entries):
        inner = outer[n // ncols, n % ncols].subgridspec(2, 1, height_ratios=(3, 1), hspace=0)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)
        top.errorbar(k, k * pk / np.pi, k * err / np.pi, fmt="o", ms=3, capsize=2, color="C0", label="data")
        top.plot(k, k * mod / np.pi, color="C1", label="model")
        top.set_title(f"{label}z={z:.1f}  $\\chi^2$={chi2:.1f}/{len(k)}", fontsize=10)
        top.tick_params(labelbottom=False)
        bottom.errorbar(k, pk / mod - 1, err / mod, fmt="o", ms=3, capsize=2, color="C0")
        bottom.axhline(0, color="C1", lw=1)
        bottom.axhspan(-0.01, 0.01, color="gray", alpha=0.2, lw=0)
        bottom.set_ylim(-0.1, 0.1)
        bottom.set_yticks([-0.05, 0, 0.05])
        top.yaxis.set_major_locator(MaxNLocator(prune="lower"))
        if n % ncols == 0:
            top.set_ylabel(r"$k P_{1D}/\pi$")
            bottom.set_ylabel("data/model - 1")
        if n >= len(entries) - ncols:
            bottom.set_xlabel(r"$k_\parallel$ [s/km]")
        spectra.append(top)
    spectra[0].legend(fontsize=8)
    prefix = "" if title is None else title + "\n"
    fig.suptitle(prefix + format_goodness_of_fit(goodness_of_fit(model, point), multiline=True))
    return fig, np.array(spectra)


def plot_star_ellipse(model, point, local_errors, ax=None, label=None, color=None,
                      levels=None, unblind=False, nsigma=(1, 2)):
    """Plot the local-error ellipses of ``(n_star, Delta2_star)`` and the minimum.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph containing a ``LyaCosmology`` component.
    point : mapping
        Physical sampled values at the minimum.
    local_errors : LocalErrors
        Local parameter covariance at ``point``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. to overlay several fits. A new figure otherwise.
    label : str, optional
        Prefix of the legend entries, useful when overlaying fits.
    color : str, optional
        Colour of this fit; levels are distinguished by line style.
    levels : sequence of float, optional
        Draw the contours enclosing these 2D probabilities instead (e.g.
        ``(0.6827, 0.9545)``: Delta chi2 = 2.30, 6.18).
    nsigma : sequence of float, default=(1, 2)
        Contours whose projection onto each axis is the 1D ``nsigma`` interval
        (Delta chi2 = nsigma**2), as cup1d's figures
        (:func:`cup1d.postprocessing.minimizer.star.contour_delta_chi2`).
        Ignored if ``levels`` is given.
    unblind : bool, default=False
        Plot values without the native DESI blinding offsets, for unblinded
        data sets such as DESI DR1 P1D. The "(blinded)" labels are then omitted.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    summary : dict
        Blinded values, marginal errors, correlation and the propagated
        :class:`PropagatedErrors`.

    Notes
    -----
    The full parameter covariance is propagated through finite-difference
    derivatives of :func:`star_values`, keeping all correlations. Unless
    ``unblind``, values carry the native DESI blinding offsets, and every legend
    entry and axis label says "(blinded)" whenever an offset applies.
    """
    import matplotlib.pyplot as plt
    from cup1d.postprocessing.minimizer.star import contour_delta_chi2, ellipse_offsets

    stars = propagate_local_errors(point, local_errors, lambda p: star_values(model, p, unblind))
    delta2, nstar = stars.values
    suffix = " (blinded)" if is_blinded(model) and not unblind else ""
    prefix = "" if label is None else label + ", "
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure
    if color is None:
        color = ax._get_lines.get_next_color()
    for contour, style in zip(_contours(levels, nsigma), ("-", "--", ":", "-.")):
        ellipse = ellipse_offsets(stars.covariance[::-1, ::-1], contour_delta_chi2(**contour[0]))  # (n_star, Delta2_star)
        ax.plot(nstar + ellipse[0], delta2 + ellipse[1], style, color=color,
                label=f"{prefix}{contour[1]}{suffix}")
    ax.plot(nstar, delta2, "x", color=color, ms=10, mew=2, label=f"{prefix}MLE{suffix}")
    ax.set_xlabel(r"$n_\star$" + suffix)
    ax.set_ylabel(r"$\Delta^2_\star$" + suffix)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    summary = {
        "Delta2_star": float(delta2), "n_star": float(nstar),
        "sigma_Delta2_star": float(stars.errors[0]), "sigma_n_star": float(stars.errors[1]),
        "correlation": float(stars.covariance[0, 1] / np.prod(stars.errors)),
        "blinded": bool(suffix), "propagated": stars,
    }
    return fig, ax, summary


def _contours(levels, nsigma):
    """``[(contour_delta_chi2 keywords, legend label), ...]`` of the ellipses."""
    if levels is not None:
        return [({"level": level}, f"{100 * level:.0f}%") for level in levels]
    return [({"nsigma": n}, f"{n:g}$\\sigma$") for n in nsigma]


def plot_vega_point(model, point, title=None):
    """Plot every Vega correlation against its data in four mu wedges.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph with a coupled Vega likelihood.
    point : mapping
        Physical sampled values.
    title : str, optional
        Prefix of the goodness-of-fit title.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per correlation, from Vega's maintained wedge plotter, using
        exactly the native parameters of the data likelihood.
    """
    vega = vega_component(model)
    if vega is None:
        raise ValueError("the model has no Vega likelihood")
    model.logposterior(point, return_derived=False)
    values = vega.parameters_for_evaluation(vega_inputs(model, vega, point))
    correlations = vega.backend.compute_model(values, run_init=False)
    prefix = "" if title is None else title + "\n"
    summary = prefix + format_goodness_of_fit(goodness_of_fit(model, point), multiline=True)
    figures = []
    for name, correlation in correlations.items():
        vega.backend.plots.plot_4wedges(models=[correlation], corr_name=name, title=None,
                                        mu_bin_labels=True, no_font=True)
        figure = vega.backend.plots.fig
        figure.suptitle(f"{name}: {summary}")
        figure.tight_layout(rect=(0, 0, 1, .88))
        figures.append(figure)
    return figures


_STAR_LABELS = {"Delta2star": r"$\Delta^2_\star$", "nstar": r"$n_\star$"}


def plot_local_triangle(model, point, local_errors, parameters=("Delta2star", "nstar", "ap", "at"),
                        axes=None, label=None, color=None, levels=None,
                        unblind=False, labels=None, nsigma=(1, 2)):
    """Triangle plot of local-error ellipses for star and sampled parameters.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph containing a ``LyaCosmology`` component.
    point : mapping
        Physical sampled values at the minimum.
    local_errors : LocalErrors
        Local parameter covariance at ``point``.
    parameters : sequence of str, default=('Delta2star', 'nstar', 'ap', 'at')
        ``'Delta2star'``/``'nstar'`` (blinded unless ``unblind``) and/or any
        sampled parameter names.
    axes : numpy.ndarray, optional
        Square axes array from a previous call, to overlay several fits.
    label, color, levels, unblind, nsigma
        As in :func:`plot_star_ellipse`.
    labels : mapping, optional
        Axis labels by parameter name; defaults to LaTeX for star parameters.

    Returns
    -------
    fig, axes
        Figure and ``(n, n)`` axes array (upper triangle hidden).
    summary : dict
        Best-fit values, marginal errors, the correlation matrix and the
        :class:`PropagatedErrors`.

    Notes
    -----
    The full parameter covariance is propagated to the requested quantities
    (sampled parameters enter with unit derivatives), keeping all correlations.
    Diagonal panels show the Gaussian marginal of each quantity.
    """
    import matplotlib.pyplot as plt

    parameters = tuple(parameters)
    stars_needed = any(name in _STAR_LABELS for name in parameters)

    def observables(p):
        stars = dict(zip(("Delta2star", "nstar"), star_values(model, p, unblind))) if stars_needed else {}
        return np.array([stars[name] if name in stars else p[name] for name in parameters])

    propagated = propagate_local_errors(point, local_errors, observables)
    values, covariance, errors = propagated.values, propagated.covariance, propagated.errors
    blinded = stars_needed and is_blinded(model) and not unblind
    labels = dict(labels or {})
    for name in parameters:
        default = _STAR_LABELS.get(name, name.replace("_", r"\_"))
        if name in _STAR_LABELS and blinded:
            default += " (blinded)"
        labels.setdefault(name, default)
    size = len(parameters)
    if axes is None:
        fig, axes = plt.subplots(size, size, figsize=(2.6 * size, 2.6 * size), squeeze=False)
        for i in range(size):
            for j in range(i + 1, size):
                axes[i, j].set_visible(False)
    else:
        fig = axes[0, 0].figure
    if color is None:
        color = axes[0, 0]._get_lines.get_next_color()
    from cup1d.postprocessing.minimizer.star import contour_delta_chi2, ellipse_offsets
    for i in range(size):
        ax = axes[i, i]
        grid = values[i] + errors[i] * np.linspace(-4, 4, 200)
        ax.plot(grid, np.exp(-0.5 * ((grid - values[i]) / errors[i]) ** 2), color=color)
        ax.set_yticks([])
        ax.set_title(f"{labels[parameters[i]]} = {values[i]:.4g} $\\pm$ {errors[i]:.2g}", fontsize=9)
        for j in range(i):
            ax = axes[i, j]
            block = covariance[np.ix_([j, i], [j, i])]
            for contour, style in zip(_contours(levels, nsigma), ("-", "--", ":", "-.")):
                ellipse = ellipse_offsets(block, contour_delta_chi2(**contour[0]))
                ax.plot(values[j] + ellipse[0], values[i] + ellipse[1], style, color=color)
            ax.plot(values[j], values[i], "x", color=color, ms=8, mew=2)
    for i in range(size):
        for j in range(i + 1):
            if i == size - 1:
                axes[i, j].set_xlabel(labels[parameters[j]])
            else:
                axes[i, j].tick_params(labelbottom=False)
            if j == 0 and i > 0:
                axes[i, j].set_ylabel(labels[parameters[i]])
            elif j > 0:
                axes[i, j].tick_params(labelleft=False)
    prefix = "" if label is None else label + ", "
    handles = [plt.Line2D([], [], color=color, ls=style, label=f"{prefix}{100 * level:.0f}%")
               for level, style in zip(levels, ("-", "--", ":", "-."))]
    handles.append(plt.Line2D([], [], color=color, marker="x", ls="", mew=2, label=f"{prefix}MLE"))
    legend_ax = axes[0, size - 1] if size > 1 else axes[0, 0]
    if size > 1:
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        previous = legend_ax.get_legend()
        if previous is not None:
            handles = previous.legend_handles + handles
    legend_ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    summary = {
        "values": dict(zip(parameters, map(float, values))),
        "errors": dict(zip(parameters, map(float, errors))),
        "correlation": covariance / np.outer(errors, errors),
        "blinded": bool(blinded), "propagated": propagated,
    }
    return fig, axes, summary
