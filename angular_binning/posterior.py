"""
Functions for plotting posterior distributions.
"""

import gaussian_cl_likelihood.python.posteriors # https://github.com/robinupham/gaussian_cl_likelihood
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.special
import scipy.stats


def cl_posts(log_like_filemask, contour_levels_sig, n_bps, colours, linestyles, ellipse_check=False,
             plot_save_path=None):
    """
    Plot w0-wa joint posteriors from the full-sky power spectrum for different numbers of bandpowers.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholder for ``{n_bp}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        n_bps (list): 2D nested list of numbers of bandpowers to plot. The top-level list defines the panels; the inner
                      lists are the different numbers of bandpowers to plot within each panel. For example,
                      ``[[30, 1], [30, 5], [30, 10]]`` will produce three panels showing 1, 5, and 10 bandpowers,
                      with all panels also showing 30 bandpowers. There must be the same number of numbers of bandpowers
                      within each panel.
        colours (list): List of matplotlib colours, corresponding to the different numbers of bandpowers within each
                        panel. All panels will use the same colours.
        linestyles (list): Like ``colours``, but matplotlib linestyles.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=3, figsize=(12.8, 4), sharey=True)
    plt.subplots_adjust(left=0.07, right=.97, wspace=0, bottom=.13, top=.98)

    # Plot each panel at a time
    for panel_idx, (a, panel_n_bps) in enumerate(zip(ax, n_bps)):
        for n_bp, colour, linestyle in zip(panel_n_bps, colours, linestyles):
            print(f'Panel {panel_idx + 1} / {len(n_bps)}, n_bp = {n_bp}')

            # Load log-likelihood
            log_like_path = log_like_filemask.format(n_bp=n_bp)
            x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like) - 0.5 * np.amin(log_like - np.amax(log_like))
            post = np.exp(log_like)

            # Form x and y grids and determine grid cell size (requires and checks for regular grid)
            x_vals_unique = np.unique(x_vals)
            dx = x_vals_unique[1] - x_vals_unique[0]
            assert np.allclose(np.diff(x_vals_unique), dx)
            y_vals_unique = np.unique(y_vals)
            dy = y_vals_unique[1] - y_vals_unique[0]
            dxdy = dx * dy
            assert np.allclose(np.diff(y_vals_unique), dy)
            x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

            # Grid posterior and convert to confidence intervals
            post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

            # Calculate contours
            cont = a.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=linestyle,
                             linewidths=2.5)

            # Fit ellipse
            for collection in cont.collections:
                paths = collection.get_paths()
                if not paths:
                    continue

                # Find biggest enclosed contour
                path_lengths = [path.vertices.shape[0] for path in paths]
                main_path = paths[np.argmax(path_lengths)]
                path_x = main_path.vertices[:, 0]
                path_y = main_path.vertices[:, 1]

                # Calculate ellipse centre using midpoint of x and y
                centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)

                # Calculate angle using linear regression
                slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
                phi = -np.arctan(slope)

                # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
                height = np.ptp(path_y) / np.cos(phi)

                # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
                # then finding range of new x
                width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

                # Draw the ellipse and hide the original
                fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width, height=height, angle=np.rad2deg(phi),
                                                         ec=collection.get_ec()[0], fc='None',
                                                         lw=collection.get_lw()[0], ls=collection.get_ls()[0])
                a.add_patch(fit_ellipse)
                if not ellipse_check:
                    collection.set_visible(False)

    # Limits
    for a in ax:
        a.set_xlim(-1.01, -0.99)
        a.set_ylim(-0.03, 0.035)

    # Axis labels
    for a in ax:
        a.set_xlabel(r'$w_0$')
    ax[0].set_ylabel(r'$w_a$')

    # Legends
    for a, panel_n_bps in zip(ax, n_bps):
        handles = [matplotlib.lines.Line2D([0], [0], lw=2.5, c=c, ls=ls[0]) for c, ls in zip(colours, linestyles)]
        labels = [f'{n_bp} bandpower{"s" if n_bp > 1 else ""}' for n_bp in panel_n_bps]
        a.legend(handles, labels, frameon=False, loc='upper right')

    # Remove overlapping tick labels
    for a in ax[1:]:
        a.set_xticks(a.get_xticks()[1:])

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def cf_posts(log_like_filemask, contour_levels_sig, n_bins, colours, linestyles, ellipse_check=False,
             plot_save_path=None):
    """
    Plot w0-wa joint posteriors from the full-sky correlation function for different numbers of theta bins.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_cf_gauss_loop_nbin``, with placeholder for ``{n_bin}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        n_bins (list): 2D nested list of numbers of theta bins to plot. The top-level list defines the panels; the
                       inner lists are the different numbers of theta bins to plot within each panel. For example,
                       ``[[30, 5], [30, 10], [30, 20]]`` will produce three panels showing 5, 10, and 20 theta bins,
                       with all panels also showing 30 bins. There must be the same number of numbers of theta bins
                       within each panel.
        colours (list): List of matplotlib colours, corresponding to the different numbers of theta bins within each
                        panel. All panels will use the same colours.
        linestyles (list): Like ``colours``, but matplotlib linestyles.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=3, figsize=(12.8, 4), sharey=True)
    plt.subplots_adjust(left=0.07, right=.97, wspace=0, bottom=.13, top=.98)

    # Plot each panel at a time
    for panel_idx, (a, panel_n_bins) in enumerate(zip(ax, n_bins)):
        for nbin, colour, linestyle in zip(panel_n_bins, colours, linestyles):
            print(f'Panel {panel_idx + 1} / {len(n_bins)}, nbin = {nbin}')

            # Load log-likelihood
            log_like_path = log_like_filemask.format(n_bin=nbin)
            x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like) - 0.5 * np.amin(log_like - np.amax(log_like))
            post = np.exp(log_like)

            # Form x and y grids and determine grid cell size (requires and checks for regular grid)
            x_vals_unique = np.unique(x_vals)
            dx = x_vals_unique[1] - x_vals_unique[0]
            assert np.allclose(np.diff(x_vals_unique), dx)
            y_vals_unique = np.unique(y_vals)
            dy = y_vals_unique[1] - y_vals_unique[0]
            dxdy = dx * dy
            assert np.allclose(np.diff(y_vals_unique), dy)
            x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

            # Grid posterior and convert to confidence intervals
            post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

            # Calculate contours
            cont = a.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=linestyle,
                             linewidths=2.5)

            # Fit ellipse
            for collection in cont.collections:
                paths = collection.get_paths()
                if not paths:
                    continue

                # Find biggest enclosed contour
                path_lengths = [path.vertices.shape[0] for path in paths]
                main_path = paths[np.argmax(path_lengths)]
                path_x = main_path.vertices[:, 0]
                path_y = main_path.vertices[:, 1]

                # Calculate ellipse centre using midpoint of x and y
                centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)

                # Calculate angle using linear regression
                slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
                phi = -np.arctan(slope)

                # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
                height = np.ptp(path_y) / np.cos(phi)

                # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
                # then finding range of new x
                width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

                # Draw the ellipse and hide the original
                fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width, height=height, angle=np.rad2deg(phi),
                                                         ec=collection.get_color()[0], fc='None',
                                                         lw=collection.get_lw()[0], ls=collection.get_ls()[0])
                a.add_patch(fit_ellipse)
                if not ellipse_check:
                    collection.set_visible(False)

    # Limits
    for a in ax:
        a.set_xlim(-1.01, -0.99)
        a.set_ylim(-0.033, 0.035)

    # Axis labels
    for a in ax:
        a.set_xlabel(r'$w_0$')
    ax[0].set_ylabel(r'$w_a$')

    # Legends
    for a, panel_n_bins in zip(ax, n_bins):
        handles = [matplotlib.lines.Line2D([0], [0], lw=2.5, c=c, ls=ls[0]) for c, ls in zip(colours, linestyles)]
        labels = [f'{nbin} $\\theta$ bins' for nbin in panel_n_bins]
        a.legend(handles, labels, frameon=False, loc='upper right')

    # Remove overlapping tick labels
    for a in ax[1:]:
        a.set_xticks(a.get_xticks()[1:])

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()
