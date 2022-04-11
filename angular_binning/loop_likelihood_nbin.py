"""
Functions to run likelihoods, automatically looping over numbers of angular bins.
"""

import glob
import os.path
import time
import warnings

import gaussian_cl_likelihood.python.simulation # https://github.com/robinupham/gaussian_cl_likelihood
import numpy as np

import angular_binning.like_bp_gauss as like_bp
import angular_binning.like_bp_gauss_mix as like_bp_mix
import angular_binning.like_cf_gauss as like_cf

warnings.filterwarnings('error') # terminate on warning


def like_bp_gauss_loop_nbin(grid_dir, n_bps, n_zbin, lmax, lmin_like, lmin_in, fid_pos_pos_dir, fid_she_she_dir,
                            fid_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, pbl_save_dir, obs_bp_save_dir,
                            inv_cov_save_dir, varied_params, like_save_dir, cov_fsky=1.0):
    """
    Run the like_bp_gauss likelihood module over a CosmoSIS grid repeatedly for different numbers of bandpowers, saving
    a separate likelihood file for each number of bandpowers.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        n_bps (list): List of numbers of bandpowers.
        n_zbin (int): Number of redshift bins.
        lmax (int): Maximum l to use in the likelihood.
        lmin_like (int): Minimum l to use in the likelihood.
        lmin_in (int): Minimum l supplied in theory power spectra.
        fid_pos_pos_dir (str): Path to fiducial position-position power spectra.
        fid_she_she_dir (str): Path to fiducial shear-shear power spectra.
        fid_pos_she_dir (str): Path to fiducial position-shear power spectra.
        pos_nl_path (str): Path to text file containing position noise power spectrum.
        she_nl_path (str): Path to text file containing shear noise power spectrum.
        noise_ell_path (str): Path to text file containing ells for the noise power spectra.
        pbl_save_dir (str): Path to directory into which to save bandpower binning matrices, which are then loaded
                            inside the likelihood module.
        obs_bp_save_dir (str): Path to directory into which to save binned 'observed' (fiducial) power spectra, which
                               are then loaded inside the likelihood module.
        inv_cov_save_dir (str): Path to directory into which to save precomputed inverse bandpower covariance matrices,
                                which are then loaded inside the likelihood module.
        varied_params (list): List of CosmoSIS parameter names whose values are varied across the grid.
        like_save_dir (str): Path to directory into which to save likelihood files, one for each number of bandpowers.
        cov_fsky (float, optional): If supplied, covariance will be multiplied by 1 / cov_fsky. (Default 1.0.)
    """

    print(f'Starting at {time.strftime("%c")}')

    # Calculate some useful quantities
    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell_like = lmax - lmin_like + 1
    ell_like = np.arange(lmin_like, lmax + 1)
    n_ell_in = lmax - lmin_in + 1
    assert lmin_in <= lmin_like

    # Load fiducial Cls
    print(f'Loading fiducial Cls at {time.strftime("%c")}')
    fid_cl = like_bp.load_spectra(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax=lmax,
                                  lmin=lmin_in)[:, lmin_like:]
    assert fid_cl.shape == (n_spec, n_ell_like)

    # Add noise
    print(f'Adding noise at {time.strftime("%c")}')
    pos_nl = np.loadtxt(pos_nl_path, max_rows=n_ell_in)[(lmin_like - lmin_in):]
    she_nl = np.loadtxt(she_nl_path, max_rows=n_ell_in)[(lmin_like - lmin_in):]
    fid_cl[:n_field:2, :] += pos_nl
    fid_cl[1:n_field:2, :] += she_nl

    # Obs = fid
    obs_cl = fid_cl

    # Precompute unbinned covariance
    cl_covs = np.full((n_ell_like, n_spec, n_spec), np.nan)
    for l in range(lmin_like, lmax + 1):
        print(f'Calculating Cl covariance l = {l} / {lmax} at {time.strftime("%c")}')
        cl_covs[l - lmin_like, :, :] = like_cf.calculate_cl_cov_l(fid_cl[:, l - lmin_like], l, n_field)
    assert np.all(np.isfinite(cl_covs))
    assert np.all([np.all(np.linalg.eigvals(cl_cov) > 0) for cl_cov in cl_covs])

    # Multiply by 1/fsky
    print(f'Multiplying covariance by 1/fsky with fsky = {cov_fsky} at {time.strftime("%c")}')
    cl_covs *= 1. / cov_fsky

    # Iterate over numbers of bandpowers
    for n_bp in n_bps:
        print(f'Starting n_bp = {n_bp} at {time.strftime("%c")}')

        # Form binning matrix
        print(f'{n_bp}bp: Forming binning matrix at {time.strftime("%c")}')
        pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bp, lmin_like, lmax)

        # Save binning matrix to disk
        pbl_path = os.path.join(pbl_save_dir, f'pbl_lmin{lmin_like}_lmax{lmax}_{n_bp}bp.txt')
        pbl_header = (f'Bandpower binning matrix output from {__file__}.like_bp_gauss_loop_nbin for '
                      f'lmin = {lmin_like}, lmax = {lmax}, n_bp = {n_bp}, at {time.strftime("%c")}')
        np.savetxt(pbl_path, pbl, header=pbl_header)
        print(f'{n_bp}bp: Saved binning matrix to {pbl_path} at {time.strftime("%c")}')

        # Bin observation
        print(f'{n_bp}bp: Binning observation at {time.strftime("%c")}')
        obs_bp = np.einsum('bl,sl->sb', pbl, obs_cl)
        assert obs_bp.shape == (n_spec, n_bp)
        assert np.all(np.isfinite(obs_bp))

        # Save observation to disk
        obs_bp_dir = os.path.join(obs_bp_save_dir, f'lmin{lmin_like}_lmax{lmax}_{n_bp}bp/')
        gaussian_cl_likelihood.python.simulation.save_cls_nob(obs_bp, n_zbin, obs_bp_dir)

        # Bin covariance (b = bandpower, l = ell, s = spec1, t = spec2)
        print(f'{n_bp}bp: Binning covariance at {time.strftime("%c")}')
        bp_covs = np.einsum('bl,lst->bst', pbl ** 2, cl_covs)
        assert bp_covs.shape == (n_bp, n_spec, n_spec)
        assert np.all([np.all(np.linalg.eigvals(bp_cov) > 0) for bp_cov in bp_covs])

        # Combine into a single covariance matrix grouped by spectrum
        print(f'{n_bp}bp: Combining covariance at {time.strftime("%c")}')
        n_data = n_spec * n_bp
        cov = np.full((n_data, n_data), np.nan)
        for spec1 in range(n_spec):
            for spec2 in range(n_spec):
                cov[(spec1 * n_bp):((spec1 + 1) * n_bp),
                    (spec2 * n_bp):((spec2 + 1) * n_bp)] = np.diag(bp_covs[:, spec1, spec2])
        assert np.all(np.isfinite(cov))
        assert np.allclose(cov, cov.T, atol=0)
        assert np.all(np.linalg.eigvals(cov) > 0)

        # Invert covariance
        print(f'{n_bp}bp: Inverting covariance at {time.strftime("%c")}')
        inv_cov = np.linalg.inv(cov)
        assert inv_cov.shape == (n_data, n_data)
        assert np.all(np.isfinite(inv_cov))
        assert np.allclose(inv_cov, inv_cov.T, atol=0, rtol=1e-4)
        assert np.all(np.linalg.eigvals(inv_cov) > 0)

        # Save inverse covariance to disk
        inv_cov_path = os.path.join(inv_cov_save_dir, f'inv_cov_{n_zbin}zbin_lmin{lmin_like}_lmax{lmax}_{n_bp}bp.npz')
        inv_cov_header = (f'Inverse bandpower covariance from {__file__}.like_bp_gauss_loop_nbin for parameters '
                          f'n_zbin = {n_zbin}, fid_pos_pos_dir = {fid_pos_pos_dir}, '
                          f'fid_she_she_dir = {fid_she_she_dir}, fid_cl_pos_she_dir = {fid_pos_she_dir}, '
                          f'pos_nl_path = {pos_nl_path}, she_nl_path = {she_nl_path}, lmax = {lmax}, '
                          f'lmin_like = {lmin_like}, n_bp = {n_bp}, at {time.strftime("%c")}')
        np.savez_compressed(inv_cov_path, inv_cov=inv_cov, header=inv_cov_header)
        print(f'{n_bp}bp: Saved inverse covariance to {inv_cov_path} at {time.strftime("%c")}')

        # Setup the likelihood module
        print(f'{n_bp}bp: Setting up likelihood module at {time.strftime("%c")}')
        obs_pos_pos_dir = os.path.join(obs_bp_dir, 'galaxy_cl/')
        obs_she_she_dir = os.path.join(obs_bp_dir, 'shear_cl/')
        obs_pos_she_dir = os.path.join(obs_bp_dir, 'galaxy_shear_cl/')
        config = like_bp.setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path,
                               noise_ell_path, pbl_path, inv_cov_path, lmax, lmin_like)
        print(f'{n_bp}bp: Setup complete at {time.strftime("%c")}')

        # Loop over every input directory
        source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
        n_dirs = len(source_dirs)
        if n_dirs == 0:
            warnings.warn(f'{n_bp}bp: No matching directories. Terminating at {time.strftime("%c")}')
            return
        n_params = len(varied_params)
        if n_params == 0:
            warnings.warn(f'{n_bp}bp: No parameters specified. Terminating at {time.strftime("%c")}')
            return
        res = []
        for i, source_dir in enumerate(source_dirs):
            print(f'{n_bp}bp: Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}')

            # Extract cosmological parameters
            params = [None]*n_params
            values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
            with open(values_path, encoding='UTF-8') as f:
                for line in f:
                    for param_idx, param in enumerate(varied_params):
                        param_str = f'{param} = '
                        if param_str in line:
                            params[param_idx] = float(line[len(param_str):])
            err_str = f'{n_bp}bp: Not all parameters in varied_params found in {values_path}'
            assert np.all([param is not None for param in params]), err_str

            # Check the ells for consistency
            theory_ell = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'), max_rows=n_ell_in)
            theory_ell = theory_ell[(lmin_like - lmin_in):]
            assert np.array_equal(theory_ell, ell_like)

            # Load theory Cls
            th_pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
            th_she_she_dir = os.path.join(source_dir, 'shear_cl/')
            th_pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
            theory_cl = like_bp.load_spectra(n_zbin, th_pos_pos_dir, th_she_she_dir, th_pos_she_dir, lmax=lmax,
                                             lmin=lmin_in)[:, lmin_like:]
            assert theory_cl.shape == (n_spec, n_ell_like)

            # Evaluate likelihood
            log_like_gauss = like_bp.execute(theory_ell, theory_cl, config)

            # Store cosmological params & likelihood
            res.append([*params, log_like_gauss])

        # Save results to file
        res_grid = np.asarray(res)
        param_names = ' '.join(varied_params)
        like_path = os.path.join(like_save_dir, f'like_lmax{lmax}_{n_bp}bp.txt')
        like_header = (f'Output from {__file__}.like_bp_gauss_loop_nbin for parameters:\ngrid_dir = {grid_dir}\n'
                       f'n_zbin = {n_zbin}\nlmax = {lmax}\nlmin_like = {lmin_like}\nlmin_in = {lmin_in}\n'
                       f'fid_pos_pos_dir = {fid_pos_pos_dir}\nfid_she_she_dir = {fid_she_she_dir}\n'
                       f'fid_pos_she_dir = {fid_pos_she_dir}\npos_nl_path = {pos_nl_path}\n'
                       f'she_nl_path = {she_nl_path}\nnoise_ell_path = {noise_ell_path}\n'
                       f'pbl_save_dir = {pbl_save_dir}\nobs_bp_save_dir = {obs_bp_save_dir}\n'
                       f'inv_cov_save_dir = {inv_cov_save_dir}\ncov_fsky = {cov_fsky}\nn_bp = {n_bp}\n'
                       f'at {time.strftime("%c")}\n\n'
                       f'{param_names} log_like')
        np.savetxt(like_path, res_grid, header=like_header)
        print(f'{n_bp}bp: Saved likelihood file to {like_path} at {time.strftime("%c")}')

        print(f'{n_bp}bp: Done at {time.strftime("%c")}')
        print()

    print(f'All done at {time.strftime("%c")}')


def like_bp_gauss_mix_loop_nbin(grid_dir, n_bps, n_zbin, lmax_like, lmin_like, lmax_in, lmin_in, fid_pos_pos_dir,
                                fid_she_she_dir, fid_pos_she_dir, pos_nl_path, she_nl_path, mixmats_path,
                                bp_cov_filemask, binmixmat_save_dir, obs_bp_save_dir, varied_params, like_save_dir):
    """
    Run the like_bp_gauss_mix likelihood module over a CosmoSIS grid repeatedly for different numbers of bandpowers,
    saving a separate likelihood file for each number of bandpowers.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        n_bps (list): List of numbers of bandpowers.
        n_zbin (int): Number of redshift bins.
        lmax_like (int): Maximum l to use in the likelihood.
        lmin_like (int): Minimum l to use in the likelihood.
        lmax_in (int): Maximum l included in mixing.
        lmin_in (int): Minimum l supplied in theory and noise power spectra.
        fid_pos_pos_dir (str): Path to fiducial position-position power spectra.
        fid_she_she_dir (str): Path to fiducial shear-shear power spectra.
        fid_pos_she_dir (str): Path to fiducial position-shear power spectra.
        pos_nl_path (str): Path to text file containing position noise power spectrum.
        she_nl_path (str): Path to text file containing shear noise power spectrum.
        mixmats_path (str): Path to mixing matrices in numpy .npz file with four arrays (mixmat_nn_to_nn,
                            mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee) each with shape
                            (lmax_like - lmin_in + 1, lmax_in - lmin_in + 1).
        bp_cov_filemask (str): Path to precomputed bandpower covariance with {n_bp} placeholder, in numpy .npz file with
                               array name cov, with shape (n_data, n_data) where n_data = n_spectra * n_bandpowers.
        binmixmat_save_dir (str): Path to directory into which to save combined mixing and binning matrices, which are
                                  then loaded inside the likelihood module.
        obs_bp_save_dir (str): Path to directory into which to save binned 'observed' (mixed fiducial) power spectra,
                               which are then loaded inside the likelihood module.
        varied_params (list): List of CosmoSIS parameter names whose values are varied across the grid.
        like_save_dir (str): Path to directory into which to save likelihood files, one for each number of bandpowers.
    """

    print(f'Starting at {time.strftime("%c")}')

    # Calculate some useful quantities
    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell_like = lmax_like - lmin_like + 1
    n_ell_in = lmax_in - lmin_in + 1
    ell_in = np.arange(lmin_in, lmax_in + 1)

    # Form list of power spectra
    print('Forming list of power spectra')
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    assert len(fields) == n_field
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    assert len(spectra) == n_spec

    # Load fiducial Cls
    print(f'Loading fiducial Cls at {time.strftime("%c")}')
    fid_cl = like_bp.load_spectra(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax_in, lmin_in)
    fid_cl = fid_cl[:, lmin_in:]
    assert fid_cl.shape == (n_spec, n_ell_in)

    # Add noise
    print(f'Adding noise at {time.strftime("%c")}')
    pos_nl = np.loadtxt(pos_nl_path, max_rows=n_ell_in)
    she_nl = np.loadtxt(she_nl_path, max_rows=n_ell_in)
    fid_cl[:n_field:2, :] += pos_nl
    fid_cl[1:n_field:2, :] += she_nl

    # Load mixing matrices
    print(f'Loading mixing matrices at {time.strftime("%c")}')
    lowl_skip = lmin_like - lmin_in
    with np.load(mixmats_path) as data:
        mixmat_nn_to_nn = data['mixmat_nn_to_nn'][lowl_skip:, :]
        mixmat_ne_to_ne = data['mixmat_ne_to_ne'][lowl_skip:, :]
        mixmat_ee_to_ee = data['mixmat_ee_to_ee'][lowl_skip:, :]
        mixmat_bb_to_ee = data['mixmat_bb_to_ee'][lowl_skip:, :]
    mixmat_shape = (n_ell_like, n_ell_in)
    assert mixmat_nn_to_nn.shape == mixmat_shape, (mixmat_nn_to_nn.shape, mixmat_shape)
    assert mixmat_ne_to_ne.shape == mixmat_shape, (mixmat_ne_to_ne.shape, mixmat_shape)
    assert mixmat_ee_to_ee.shape == mixmat_shape, (mixmat_ee_to_ee.shape, mixmat_shape)
    assert mixmat_bb_to_ee.shape == mixmat_shape, (mixmat_bb_to_ee.shape, mixmat_shape)

    # Pre-calculate mixed BB -> EE noise contribution for auto-spectra
    print(f'Mixing noise at {time.strftime("%c")}')
    cl_bb_to_ee = mixmat_bb_to_ee @ she_nl

    # Apply mixing matrices to fiducial Cls to get an unbinned 'observation'
    obs_cl = np.full((n_spec, n_ell_like), np.nan)
    for spec_idx, (f1, z1, f2, z2) in enumerate(spectra):

        print(f'Mixing {spec_idx + 1} / {n_spec} at {time.strftime("%c")}')

        # Identify the right mixing matrix
        spec_type = f1 + f2
        if spec_type == 'NN':
            mixmat = mixmat_nn_to_nn
        elif spec_type in ('NE', 'EN'):
            mixmat = mixmat_ne_to_ne
        elif spec_type == 'EE':
            mixmat = mixmat_ee_to_ee
        else:
            raise ValueError(f'Unexpected spec_type {spec_type}')

        # Apply it: l = l_out, p = l' = l_in
        cls_unmixed = fid_cl[spec_idx, :]
        cls_mixed = np.einsum('lp,p->l', mixmat, cls_unmixed)

        # Add BB->EE noise contribution for EE auto-spectra
        if spec_type == 'EE' and z1 == z2:
            cls_mixed += cl_bb_to_ee

        # Store
        obs_cl[spec_idx, :] = cls_mixed
    assert np.all(np.isfinite(obs_cl))

    # Iterate over numbers of bandpowers
    for n_bp in n_bps:
        print(f'Starting n_bp = {n_bp} at {time.strftime("%c")}')

        # Form binning matrix
        print(f'{n_bp}bp: Forming binning matrix at {time.strftime("%c")}')
        pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bp, lmin_like, lmax_like)
        if pbl.ndim == 1:
            pbl = pbl[np.newaxis, :]
        assert pbl.shape == (n_bp, n_ell_like)

        # Form combined binning and mixing matrices
        print(f'{n_bp}bp: Forming combined binning and mixing matrices at {time.strftime("%c")}')
        binmix_nn_to_nn = pbl @ mixmat_nn_to_nn
        binmix_ne_to_ne = pbl @ mixmat_ne_to_ne
        binmix_ee_to_ee = pbl @ mixmat_ee_to_ee
        binmix_bb_to_ee = pbl @ mixmat_bb_to_ee
        binmix_shape = (n_bp, n_ell_in)
        assert binmix_nn_to_nn.shape == binmix_shape
        assert binmix_ne_to_ne.shape == binmix_shape
        assert binmix_ee_to_ee.shape == binmix_shape
        assert binmix_bb_to_ee.shape == binmix_shape

        # Save combined binning and mixing matrices to disk
        bmm_filename = f'binmix_lminin{lmin_in}_lmaxin{lmax_in}_lminlike{lmin_like}_lmaxlike{lmax_like}_{n_bp}bp.npz'
        binmixmat_path = os.path.join(binmixmat_save_dir, bmm_filename)
        binmixmat_header = (f'Combined binning and mixing matrices output by {__file__} for '
                            f'mixmats_path = {mixmats_path}, lmin_in = {lmin_in}, lmax_in = {lmax_in}, '
                            f'lmin_like = {lmin_like}, lmax_like = {lmax_like}, n_bp = {n_bp}, '
                            f'at {time.strftime("%c")}')
        np.savez_compressed(binmixmat_path, binmix_tt_to_tt=binmix_nn_to_nn, binmix_te_to_te=binmix_ne_to_ne,
                            binmix_ee_to_ee=binmix_ee_to_ee, binmix_bb_to_ee=binmix_bb_to_ee, header=binmixmat_header)
        print(f'{n_bp}bp: Saved combined binnning and mixing matrices to {binmixmat_path} at {time.strftime("%c")}')

        # Bin observation
        print(f'{n_bp}bp: Binning observation at {time.strftime("%c")}')
        obs_bp = np.einsum('bl,sl->sb', pbl, obs_cl)
        assert obs_bp.shape == (n_spec, n_bp)
        assert np.all(np.isfinite(obs_bp))

        # Save observation to disk
        obs_bp_path = os.path.join(obs_bp_save_dir, f'obs_{n_bp}bp.npz')
        obs_bp_header = (f'Observed bandpowers output by {__file__} for fid_pos_pos_dir = {fid_pos_pos_dir}, '
                         f'fid_she_she_dir = {fid_she_she_dir}, fid_pos_she_dir = {fid_pos_she_dir}, '
                         f'mixmats_path = {mixmats_path}, lmin_like = {lmin_like}, lmax_like = {lmax_like}, '
                         f'n_bp = {n_bp}, at {time.strftime("%c")}')
        np.savez_compressed(obs_bp_path, obs_bp=obs_bp, header=obs_bp_header)
        print(f'{n_bp}bp: Saved binned observation to {obs_bp_path} at {time.strftime("%c")}')

        # Setup the likelihood module
        print(f'{n_bp}bp: Setting up likelihood module at {time.strftime("%c")}')
        bp_cov_path = bp_cov_filemask.format(n_bp=n_bp)
        config = like_bp_mix.setup(obs_bp_path, binmixmat_path, lmin_in, bp_cov_path, pos_nl_path, she_nl_path, lmin_in,
                                   lmax_in, n_zbin)
        print(f'{n_bp}bp: Setup complete at {time.strftime("%c")}')

        # Loop over every input directory
        source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
        n_dirs = len(source_dirs)
        if n_dirs == 0:
            warnings.warn(f'{n_bp}bp: No matching directories. Terminating at {time.strftime("%c")}')
            return
        n_params = len(varied_params)
        if n_params == 0:
            warnings.warn(f'{n_bp}bp: No parameters specified. Terminating at {time.strftime("%c")}')
            return
        res = []
        for i, source_dir in enumerate(source_dirs):
            print(f'{n_bp}bp: Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}')

            # Extract cosmological parameters
            params = [None]*n_params
            values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
            with open(values_path, encoding='ascii') as f:
                for line in f:
                    for param_idx, param in enumerate(varied_params):
                        param_str = f'{param} = '
                        if param_str in line:
                            params[param_idx] = float(line[len(param_str):])
            err_str = f'{n_bp}bp: Not all parameters in varied_params found in {values_path}'
            assert np.all([param is not None for param in params]), err_str

            # Check the ells for consistency
            galaxy_ell = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))[:n_ell_in]
            shear_ell = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))[:n_ell_in]
            galaxy_shear_ell = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))[:n_ell_in]
            assert np.array_equal(galaxy_ell, ell_in)
            assert np.array_equal(shear_ell, ell_in)
            assert np.array_equal(galaxy_shear_ell, ell_in)

            # Load theory Cls
            th_pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
            th_she_she_dir = os.path.join(source_dir, 'shear_cl/')
            th_pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
            theory_cl = like_bp_mix.load_cls(n_zbin, th_pos_pos_dir, th_she_she_dir, th_pos_she_dir, lmax=lmax_in)

            # Evaluate likelihood
            log_like_gauss = like_bp_mix.execute(theory_cl, lmin_in, config)

            # Store cosmological params & likelihood
            res.append([*params, log_like_gauss])

        # Save results to file
        res_grid = np.asarray(res)
        param_names = ' '.join(varied_params)
        like_path = os.path.join(like_save_dir, f'like_lmaxlike{lmax_like}_{n_bp}bp.txt')
        like_header = (f'Output from {__file__}.like_bp_gauss_mix_loop_nbin for parameters:\ngrid_dir = {grid_dir}\n'
                       f'n_zbin = {n_zbin}\nlmax_like = {lmax_like}\nlmin_like = {lmin_like}\nlmax_in = {lmax_in}\n'
                       f'lmin_in = {lmin_in}\nfid_pos_pos_dir = {fid_pos_pos_dir}\n'
                       f'fid_she_she_dir = {fid_she_she_dir}\nfid_pos_she_dir = {fid_pos_she_dir}\n'
                       f'pos_nl_path = {pos_nl_path}\nshe_nl_path = {she_nl_path}\nmixmats_path = {mixmats_path}\n'
                       f'bp_cov_filemask = {bp_cov_filemask}\nn_bp = {n_bp}\nat {time.strftime("%c")}\n\n'
                       f'{param_names} log_like_gauss')
        np.savetxt(like_path, res_grid, header=like_header)
        print(f'{n_bp}bp: Saved likelihood file to {like_path} at {time.strftime("%c")}')

        print(f'{n_bp}bp: Done at {time.strftime("%c")}')
        print()

    print(f'All done at {time.strftime("%c")}')


def like_cf_gauss_loop_nbin(grid_dir, n_theta_bins, n_zbin, lmin, lmax, theta_min, theta_max, fid_pos_pos_dir,
                            fid_she_she_dir, fid_pos_she_dir, obs_path, survey_area_sqdeg, gals_per_sqarcmin_per_zbin,
                            sigma_e, varied_params, like_save_dir, cov_fsky=1.0):
    """
    Run the like_cf_gauss likelihood module over a CosmoSIS grid repeatedly for different numbers of theta bins,
    saving a separate likelihood file for each number of theta bins.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        n_theta_bins (list): List of numbers of theta bins.
        n_zbin (int): Number of redshift bins.
        lmin (int): Minimum l to use as input.
        lmax (int): Maximum l to use as input.
        theta_min (float): Minimum theta in radians.
        theta_max (float): Maximum theta in radians.
        fid_pos_pos_dir (str): Path to fiducial position-position power spectra.
        fid_she_she_dir (str): Path to fiducial shear-shear power spectra.
        fid_pos_she_dir (str): Path to fiducial position-shear power spectra.
        obs_path (str): Path to mock observation generated by obs_from_fid.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin_per_zbin (float): Average number of galaxies per square arcminute per redshift bin.
        sigma_e (float): Intrinsic ellipticity dispersion per component.
        varied_params (list): List of CosmoSIS parameter names whose values are varied across the grid.
        like_save_dir (str): Path to directory into which to save likelihood files, one for each number of bandpowers.
        cov_fsky (float, optional): Sky fraction, default 1. Covariance is multiplied by 1/cov_fsky.
    """

    print(f'Starting at {time.strftime("%c")}')

    # Calculate some useful quantities
    ell = np.arange(lmin, lmax + 1)
    n_ell = lmax - lmin + 1

    # Iterate over number of theta bins
    for n_bin_idx, n_bin in enumerate(n_theta_bins):
        print(f'Starting n_bin = {n_bin} at {time.strftime("%c")}')

        # Setup the likelihood module
        print(f'{n_bin} bins: Setting up likelihood module at {time.strftime("%c")}')
        if n_bin_idx == 0:
            cl_covs_args = {'cl_covs': None, 'return_cl_covs': True}
        else:
            cl_covs_args = {'cl_covs': cl_covs, 'return_cl_covs': False}
        setup_output = like_cf.setup(n_zbin, obs_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax,
                                     theta_min, theta_max, n_bin, survey_area_sqdeg, gals_per_sqarcmin_per_zbin,
                                     sigma_e, cov_fsky=cov_fsky, **cl_covs_args)
        if n_bin_idx == 0:
            config, cl_covs = setup_output
        else:
            config = setup_output
        print(f'{n_bin} bins: Setup complete at {time.strftime("%c")}')

        # Loop over every input directory
        source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
        n_dirs = len(source_dirs)
        if n_dirs == 0:
            warnings.warn(f'{n_bin} bins: No matching directories. Terminating at {time.strftime("%c")}')
            return
        n_params = len(varied_params)
        if n_params == 0:
            warnings.warn(f'{n_bin} bins: No parameters specified. Terminating at {time.strftime("%c")}')
            return
        res = []
        for i, source_dir in enumerate(source_dirs):
            print(f'{n_bin} bins: Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}')

            # Extract cosmological parameters
            params = [None]*n_params
            values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
            with open(values_path, encoding='ascii') as f:
                for line in f:
                    for param_idx, param in enumerate(varied_params):
                        param_str = f'{param} = '
                        if param_str in line:
                            params[param_idx] = float(line[len(param_str):])
            err_str = f'{n_bin} bins: Not all parameters in varied_params found in {values_path}'
            assert np.all([param is not None for param in params]), err_str

            # Check the ells for consistency
            galaxy_ell = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))[:n_ell]
            shear_ell = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))[:n_ell]
            galaxy_shear_ell = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))[:n_ell]
            assert np.array_equal(galaxy_ell, ell)
            assert np.array_equal(shear_ell, ell)
            assert np.array_equal(galaxy_shear_ell, ell)

            # Load theory Cls
            pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
            she_she_dir = os.path.join(source_dir, 'shear_cl/')
            pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
            theory_cls, _ = like_cf.load_cls_zerob(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in=lmin)

            # Evaluate likelihood
            log_like_gauss = like_cf.execute(theory_cls, config)

            # Store cosmological params & likelihood
            res.append([*params, log_like_gauss])

        # Save results to file
        res_grid = np.asarray(res)
        param_names = ' '.join(varied_params)
        theta_min_deg = np.degrees(theta_min)
        like_save_path = os.path.join(like_save_dir, f'like_thetamin{theta_min_deg}_{n_bin}bins.txt')
        like_header = (f'Output from {__file__}.like_cf_gauss_loop_nbin for parameters:\ngrid_dir = {grid_dir}\n'
                       f'n_zbin = {n_zbin}\nlmin = {lmin}\nlmax = {lmax}\ntheta_min = {theta_min}\n'
                       f'theta_max = {theta_max}\nfid_pos_pos_dir = {fid_pos_pos_dir}\n'
                       f'fid_she_she_dir = {fid_she_she_dir}\nfid_pos_she_dir = {fid_pos_she_dir}\n'
                       f'obs_path = {obs_path}\nsurveyarea_sqdeg = {survey_area_sqdeg}\n'
                       f'gals_per_sqarcmin_per_zbin = {gals_per_sqarcmin_per_zbin}\nsigma_e = {sigma_e}\n'
                       f'cov_fsky = {cov_fsky}\nat {time.strftime("%c")}\n\n'
                       f'{param_names} log_like_gauss')
        np.savetxt(like_save_path, res_grid, header=like_header)
        print(f'{n_bin} bins: Saved likelihood file to {like_save_path} at {time.strftime("%c")}')

        print(f'{n_bin} bins: Done at {time.strftime("%c")}')
        print()

    print(f'All done at {time.strftime("%c")}')


def obs_from_fid(input_dir, output_path, n_zbin, lmax, lmin):
    """
    Produce a noiseless mock observation for input to like_cf_gauss_loop_nbin from fiducial Cls, assuming zero B-modes.

    Args:
        input_dir (str): Path to base directory containing fiducial Cls, assumed to contain subdirectories called
                         'galaxy_cl', 'shear_cl' and 'galaxy_shear_cl'.
        output_path (str): Path to save the output as a .npz file.
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        lmax (int): Maximum l to load.
        lmin (int): Minimum l in the input.
    """

    # Calculate some useful quantities
    n_field = 3 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell = lmax + 1

    # Load Cls, padded from l=0 with zeros and using zero B-modes
    pos_pos_dir = os.path.join(input_dir, 'galaxy_cl/')
    she_she_dir = os.path.join(input_dir, 'shear_cl/')
    pos_she_dir = os.path.join(input_dir, 'galaxy_shear_cl/')
    fid_cl, _ = like_cf.load_cls_zerob(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin)
    assert fid_cl.shape == (n_spec, n_ell), (fid_cl.shape, (n_spec, n_ell))

    # Save to disk in format expected as the 'observation' for like_cf_gauss_loop_nbin
    ell = np.arange(lmax + 1)
    header = (f'Output from {__file__}.obs_from_fid for input_dir = {input_dir}, n_zbin = {n_zbin}, lmax = {lmax}, '
              f'lmin = {lmin}, at {time.strftime("%c")}. All spectra including B-modes in diagonal-major order.')
    np.savez_compressed(output_path, ell=ell, obs_cls=fid_cl, header=header)
    print('Saved ' + output_path)
