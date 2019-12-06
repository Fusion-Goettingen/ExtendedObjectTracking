"""
Author: Kolja Thormann

Contains test cases for different ellipse fusion methods
"""

import numpy as np

from FusionMethods.ellipse_fusion_methods import mmsr_mc_update, regular_update, mwdp_update, rm_mean_update,\
    mmsr_pf_update
from FusionMethods.ellipse_fusion_support import sample_m, get_ellipse_params, single_particle_approx_gaussian,\
    barycenter, to_matrix
from FusionMethods.error_and_plotting import gauss_wasserstein, square_root_distance, plot_error_bars,\
    plot_convergence, plot_ellipses
from FusionMethods.constants import *


state_dtype = np.dtype([
    ('x', 'f4', 5),           # [m1, m2, al, l ,w] or [m1, m2, s11, s12, s22]
    ('cov', 'f4', (5, 5)),    # covariance
    ('error', 'O'),           # length depends on number of steps
    ('shape', 'f4', (2, 2)),  # for mean of shape matrix only
    ('weights', 'O'),         # for particle filter only, length depends on number of particles
    ('gamma', 'i4'),          # for RM mean, keep track of number of measurements
    ('name', 'O'),            # name of the fusion method
    ('color', 'O'),           # color for error plotting
])


def test_convergence_pos(steps, runs, prior, cov_prior, cov_a, cov_b, n_particles, n_particles_pf, save_path):
    """
    Test convergence of error for different fusion methods. Creates plot of root mean square error convergence and
    errors at first and last measurement step. If a uniform prior is given for alpha, ength, and width, the particle
    based methods use it while the others use the Gaussian prior (assumed to be a Gaussian approximation of the uniform
    prior). In either case, the position is still Gaussian.
    :param steps:           Number of measurements
    :param runs:            Number of MC runs
    :param prior:           Prior prediction mean (ground truth will be drawn from it each run)
    :param cov_prior:       Prior prediction covariance (ground truth will be drawn from it each run)
    :param cov_a:           Noise of sensor A
    :param cov_b:           Noise of sensor B
    :param n_particles:     Number of particles for MMGW-MC
    :param n_particles_pf:  Number of particles for MMGW-PF
    :param save_path:       Path for saving figures
    """
    error = np.zeros(steps*2)

    # setup state for various ellipse fusion methods
    mmsr_mc = np.zeros(1, dtype=state_dtype)
    mmsr_mc[0]['error'] = error.copy()
    mmsr_mc[0]['name'] = 'MMSR-MC'
    mmsr_mc[0]['color'] = 'lightgreen'
    mmsr_pf = np.zeros(1, dtype=state_dtype)
    mmsr_pf[0]['error'] = error.copy()
    mmsr_pf[0]['name'] = 'MMSR-PF'
    mmsr_pf[0]['color'] = 'darkgreen'
    regular = np.zeros(1, dtype=state_dtype)
    regular[0]['error'] = error.copy()
    regular[0]['name'] = 'Regular'
    regular[0]['color'] = 'red'
    mwdp = np.zeros(1, dtype=state_dtype)
    mwdp[0]['error'] = error.copy()
    mwdp[0]['name'] = 'MWDP'
    mwdp[0]['color'] = 'darkcyan'
    rm_mean = np.zeros(1, dtype=state_dtype)
    rm_mean[0]['error'] = error.copy()
    rm_mean[0]['name'] = 'RM Mean'
    rm_mean[0]['color'] = 'orange'

    for r in range(runs):
        print('Run %i of %i' % (r+1, runs))
        # initialize ===================================================================================================
        # create gt from prior
        gt = sample_m(prior, cov_prior, False, 1)

        # get prior in square root space
        mmsr_mc[0]['x'], mmsr_mc[0]['cov'], particles_mc = single_particle_approx_gaussian(prior, cov_prior,
                                                                                           n_particles, False)

        # get prior for regular state
        regular[0]['x'] = prior.copy()
        regular[0]['cov'] = cov_prior.copy()

        # get prior for MWDP
        mwdp[0]['x'] = prior.copy()
        mwdp[0]['cov'] = cov_prior.copy()

        # get prior for RM mean
        rm_mean[0]['x'] = prior.copy()
        rm_mean[0]['shape'] = to_matrix(prior[AL], prior[L], prior[W], False)
        rm_mean[0]['cov'] = cov_prior.copy()
        rm_mean[0]['gamma'] = 1

        # get prior for particle filter
        mmsr_pf[0]['x'], mmsr_pf[0]['cov'], particles_pf = single_particle_approx_gaussian(prior, cov_prior,
                                                                                           n_particles_pf, False)
        mmsr_pf[0]['weights'] = np.ones(n_particles_pf) / n_particles_pf

        # test different methods
        for i in range(steps):
            if i % 10 == 0:
                print('Step %i of %i' % (i + 1, steps))
            plot_cond = (r + 1 == runs) & (i + 1 == steps)

            # create measurement from gt (using alternating sensors) ===================================================
            if (i % 2) == 0:
                meas = sample_m(gt, cov_b, True, 1)
                cov_meas = cov_b.copy()
            else:
                meas = sample_m(gt, cov_a, False, 1)
                cov_meas = cov_a.copy()

            # fusion methods ===========================================================================================
            mmsr_mc_update(mmsr_mc[0], meas, cov_meas, n_particles, gt, i, steps, plot_cond, save_path, False)

            regular_update(regular[0], meas, cov_meas, gt, i, steps, plot_cond, save_path)

            mwdp_update(mwdp[0], meas, cov_meas, gt, i, steps, plot_cond, save_path)

            rm_mean_update(rm_mean[0], meas, cov_meas, gt, i, steps, plot_cond, save_path)

            mmsr_pf_update(mmsr_pf[0], meas, cov_meas, particles_pf, n_particles_pf, gt, i, steps, plot_cond, save_path,
                           False)

    mmsr_mc[0]['error'] = np.sqrt(mmsr_mc[0]['error'] / runs)
    mmsr_pf[0]['error'] = np.sqrt(mmsr_pf[0]['error'] / runs)
    regular[0]['error'] = np.sqrt(regular[0]['error'] / runs)
    mwdp[0]['error'] = np.sqrt(mwdp[0]['error'] / runs)
    rm_mean[0]['error'] = np.sqrt(rm_mean[0]['error'] / runs)

    print(mmsr_pf['error'])
    print(rm_mean['error'])

    # error plotting ===================================================================================================
    plot_error_bars(np.block([regular, rm_mean, mmsr_mc, mwdp, mmsr_pf]), steps)
    plot_convergence(np.block([regular, rm_mean, mmsr_mc, mwdp, mmsr_pf]), steps, save_path)


def test_mean(orig, cov, n_particles, save_path):
    """
    Compare the mean in original state space with mean in square root space (via MC approximation) in regards of their
    GW and SR error
    :param orig:        Mean in original state space
    :param cov:         Covariance for original state space
    :param n_particles: Number of particles for MC approximation of SR space mean
    :param save_path:   Path for saving figures
    """
    # approximate mean in SR space
    vec_mmsr, var_A, vec_particle = single_particle_approx_gaussian(orig, cov, n_particles, True)
    mat_mmsr = np.array([
        [vec_mmsr[2], vec_mmsr[3]],
        [vec_mmsr[3], vec_mmsr[4]]
    ])
    mat_mmsr = np.dot(mat_mmsr, mat_mmsr)
    l_mmsr, w_mmsr, al_mmsr = get_ellipse_params(mat_mmsr)

    # approximate mean in matrix space
    mat_mat = np.zeros((2, 2))
    vec_mat = np.zeros(2)
    for i in range(len(vec_particle)):
        vec_mat += vec_particle[i, :2]
        mat = np.array([
            [vec_particle[i, 2], vec_particle[i, 3]],
            [vec_particle[i, 3], vec_particle[i, 4]]
        ])
        mat_mat += np.dot(mat, mat)
    vec_mat /= len(vec_particle)
    mat_mat /= len(vec_particle)
    l_mat, w_mat, al_mat = get_ellipse_params(mat_mat)

    # caclulate Barycenter using optimization
    covs_sr = np.zeros((n_particles, 2, 2))
    covs_sr[:, 0, 0] = vec_particle[:, 2]
    covs_sr[:, 0, 1] = vec_particle[:, 3]
    covs_sr[:, 1, 0] = vec_particle[:, 3]
    covs_sr[:, 1, 1] = vec_particle[:, 4]
    covs = np.einsum('xab, xbc -> xac', covs_sr, covs_sr)
    bary_particles = np.zeros((n_particles, 5))
    bary_particles[:, M] = vec_particle[:, M]
    bary_particles[:, 2] = covs[:, 0, 0]
    bary_particles[:, 3] = covs[:, 0, 1]
    bary_particles[:, 4] = covs[:, 1, 1]
    bary = barycenter(bary_particles, np.ones(n_particles) / n_particles, n_particles, particles_sr=vec_particle)
    mat_mmgw = np.array([
        [bary[2], bary[3]],
        [bary[3], bary[4]],
    ])
    l_mmgw, w_mmgw, al_mmgw = get_ellipse_params(mat_mmgw)

    # approximate error
    error_orig_gw = 0
    error_orig_sr = 0
    error_mmsr_gw = 0
    error_mmsr_sr = 0
    error_mat_gw = 0
    error_mat_sr = 0
    error_mmgw_gw = 0
    error_mmgw_sr = 0
    for i in range(n_particles):
        mat_particle = np.array([
            [vec_particle[i, 2], vec_particle[i, 3]],
            [vec_particle[i, 3], vec_particle[i, 4]]
        ])
        mat_particle = np.dot(mat_particle, mat_particle)
        l_particle, w_particle, al_particle = get_ellipse_params(mat_particle)
        error_orig_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, orig[M], orig[L],
                                           orig[W], orig[AL])
        error_orig_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, orig[M],
                                              orig[L], orig[W], orig[AL])
        error_mmsr_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mmsr[M],
                                           l_mmsr, w_mmsr, al_mmsr)
        error_mmsr_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mmsr[M],
                                              l_mmsr, w_mmsr, al_mmsr)
        error_mat_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mat[M],
                                           l_mat, w_mat, al_mat)
        error_mat_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mat[M],
                                              l_mat, w_mat, al_mat)
        error_mmgw_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, bary[M], l_mmgw,
                                           w_mmgw, al_mmgw)
        error_mmgw_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, bary[M], l_mmgw,
                                              w_mmgw, al_mmgw)
    error_orig_gw = np.sqrt(error_orig_gw / n_particles)
    error_orig_sr = np.sqrt(error_orig_sr / n_particles)
    error_mmsr_gw = np.sqrt(error_mmsr_gw / n_particles)
    error_mmsr_sr = np.sqrt(error_mmsr_sr / n_particles)
    error_mat_gw = np.sqrt(error_mat_gw / n_particles)
    error_mat_sr = np.sqrt(error_mat_sr / n_particles)
    error_mmgw_gw = np.sqrt(error_mmgw_gw / n_particles)
    error_mmgw_sr = np.sqrt(error_mmgw_sr / n_particles)

    plot_ellipses(vec_mmsr[M], l_mmsr, w_mmsr, al_mmsr, orig[M], orig[L], orig[W], orig[AL], bary[M], l_mmgw, w_mmgw,
                  al_mmgw, vec_mat[M], l_mat, w_mat, al_mat, 'MMSE Estimates', save_path + 'mmse.svg',
                  est_color='green')

    # print error
    print('RMGW of original:')
    print(error_orig_gw)
    print('RMSR of original:')
    print(error_orig_sr)
    print('RMGW of mmsr:')
    print(error_mmsr_gw)
    print('RMSR of mmsr:')
    print(error_mmsr_sr)
    print('RMGW of Euclidean mmse:')
    print(error_mat_gw)
    print('RMSR of Euclidean mmse:')
    print(error_mat_sr)
    print('RMGW of mmgw_bary:')
    print(error_mmgw_gw)
    print('RMSR of mmgw_bary:')
    print(error_mmgw_sr)
