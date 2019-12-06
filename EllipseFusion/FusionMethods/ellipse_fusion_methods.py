"""
Author: Kolja Thormann

Contains various ellipse fusion methods
"""

import numpy as np

from FusionMethods.ellipse_fusion_support import particle_filter, mwdp_fusion, get_ellipse_params,\
    get_ellipse_params_from_sr, single_particle_approx_gaussian, to_matrix
from FusionMethods.error_and_plotting import error_and_plotting
from FusionMethods.constants import *


def mmsr_mc_update(mmsr_mc, meas, cov_meas, n_particles, gt, i, steps, plot_cond, save_path, use_pos):
    """
    Fusion using MMSR-MC; creates particle density in square root space of measurements and approximates it as a
    Gaussian distribution to fuse it with the current estimate in Kalman fashion
    :param mmsr_mc:     Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param n_particles: Number of particles used for approximating the transformed density
    :param gt:          Ground truth
    :param i:           Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    :param use_pos:     If false, use particles only for shape parameters and fuse position in Kalman fashion
    """
    # convert measurement
    meas_sr, cov_meas_sr, particles_meas = single_particle_approx_gaussian(meas, cov_meas, n_particles, use_pos)

    # store prior for plotting
    m_prior = mmsr_mc['x'][M]
    l_prior, w_prior, al_prior = get_ellipse_params_from_sr(mmsr_mc['x'][SR])

    # Kalman fusion
    S = mmsr_mc['cov'] + cov_meas_sr
    K = np.dot(mmsr_mc['cov'], np.linalg.inv(S))
    mmsr_mc['x'] = mmsr_mc['x'] + np.dot(K, meas_sr - mmsr_mc['x'])
    mmsr_mc['cov'] = mmsr_mc['cov'] - np.dot(np.dot(K, S), K.T)

    # save error and plot estimate
    l_post_sr, w_post_sr, al_post_sr = get_ellipse_params_from_sr(mmsr_mc['x'][SR])
    mmsr_mc['error'][i::steps] += error_and_plotting(mmsr_mc['x'][M], l_post_sr, w_post_sr, al_post_sr, m_prior,
                                                     l_prior, w_prior, al_prior, meas[M], meas[L], meas[W], meas[AL],
                                                     gt[M], gt[L], gt[W], gt[AL], plot_cond, 'MC Approximated Fusion',
                                                     save_path + 'exampleMCApprox%i.svg' % i, est_color='green')


def regular_update(regular, meas, cov_meas, gt, i, steps, plot_cond, save_path):
    """
    Fuse estimate and measurement in original state space in Kalman fashion
    :param regular:     Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param gt:          Ground truth
    :param i:           Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    """
    # store prior for plotting
    all_prior = regular['x'].copy()

    # Kalman fusion
    S = regular['cov'] + cov_meas
    K = np.dot(regular['cov'], np.linalg.inv(S))
    regular['x'] = regular['x'] + np.dot(K, meas - regular['x'])
    regular['cov'] = regular['cov'] - np.dot(np.dot(K, S), K.T)

    # save error and plot estimate
    regular['error'][i::steps] += error_and_plotting(regular['x'][M], regular['x'][L], regular['x'][W],
                                                     regular['x'][AL], all_prior[M], all_prior[L], all_prior[W],
                                                     all_prior[AL], meas[M], meas[L], meas[W], meas[AL], gt[M], gt[L],
                                                     gt[W], gt[AL], plot_cond, 'Fusion of Original State',
                                                     save_path + 'exampleRegFus%i.svg' % i)


def mwdp_update(mwdp, meas, cov_meas, gt, i, steps, plot_cond, save_path):
    """
    Fuse using MWDP; use likelihood to determine representation of ellipse in original state space with
    smallest Euclidean distance weighted by uncertainties and fuse original state representations in Kalman fashion
    using best representation
    :param mwdp:        Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param gt:          Ground truth
    :param i:           Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    """
    # store prior for plotting
    all_prior = mwdp['x'].copy()

    # Kalman fusion using best representation
    [mwdp['x'], mwdp['cov'], k] = mwdp_fusion(mwdp['x'], mwdp['cov'], meas, cov_meas)

    # save error and plot estimate
    mwdp['error'][i::steps] += error_and_plotting(mwdp['x'][M], mwdp['x'][L], mwdp['x'][W], mwdp['x'][AL], all_prior[M],
                                                  all_prior[L], all_prior[W], all_prior[AL], meas[M], meas[L], meas[W],
                                                  meas[AL], gt[M], gt[L], gt[W], gt[AL], plot_cond, 'Best Params',
                                                  save_path + 'exampleBestParams%i.svg' % i)


def rm_mean_update(rm_mean, meas, cov_meas, gt, i, steps, plot_cond, save_path):
    """
    Treat ellipse estimates as random matrices having received an equal number of measurements and fuse as proposed by
    K.  Granström  and  U.  Orguner,  “On  Spawning  and  Combination  of Extended/Group Targets Modeled With Random
    Matrices,” IEEE Transactions on Signal Processing, vol. 61, no. 3, pp. 678–692, 2013.
    :param rm_mean:     Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space (only m is used)
    :param cov_meas:    Covariance of measurement in original state space (only m is used)
    :param gt:          Ground truth
    :param i:           Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    """
    # convert measurement
    shape_meas = to_matrix(meas[AL], meas[L], meas[W], False)

    # store prior for plotting
    m_prior = rm_mean['x'][M]
    l_prior, w_prior, al_prior = get_ellipse_params(rm_mean['shape'])

    # Kalman fusion
    S_k = rm_mean['cov'][:2, :2] + cov_meas[:2, :2]
    K_k = np.dot(rm_mean['cov'][:2, :2], np.linalg.inv(S_k))
    rm_mean['x'][M] = rm_mean['x'][M] + np.dot(K_k, meas[M] - rm_mean['x'][M])
    rm_mean['cov'][:2, :2] = rm_mean['cov'][:2, :2] - np.dot(np.dot(K_k, S_k), K_k.T)
    rm_mean['shape'] = (rm_mean['gamma'] * rm_mean['shape'] + shape_meas) / (rm_mean['gamma'] + 1.0) \
                       + (rm_mean['gamma'] / (rm_mean['gamma'] + 1.0)**2) * np.outer(m_prior-meas[M], m_prior-meas[M])
    rm_mean['gamma'] += 1

    # save error and plot estimate
    l_post, w_post, al_post = get_ellipse_params(rm_mean['shape'])
    rm_mean['error'][i::steps] += error_and_plotting(rm_mean['x'][M], l_post, w_post, al_post, m_prior, l_prior,
                                                     w_prior, al_prior, meas[M], meas[L], meas[W], meas[AL], gt[M],
                                                     gt[L], gt[W], gt[AL], plot_cond, 'RM Mean',
                                                     save_path + 'exampleRMMean%i.svg' % i)


def mmsr_pf_update(mmsr_pf, meas, cov_meas, particles_pf, n_particles_pf, gt, i, steps, plot_cond, save_path, use_pos):
    """
    Fuse using MMSR-PF; keep estimate in square root space as particle density and update the weights over time; for the
    likelihood, the particles are transformed back and the sum of the likelihoods for all 4 possible representations is
    used
    :param mmsr_pf:         Current estimate (also stores error); will be modified as a result
    :param meas:            Measurement in original state space
    :param cov_meas:        Covariance of measurement in original state space
    :param particles_pf:    The particles of the particle filter in square root space
    :param n_particles_pf:  The number of particles
    :param gt:              Ground truth
    :param i:               Current measurement step
    :param steps:           Total measurement steps
    :param plot_cond:       Boolean determining whether to plot the current estimate
    :param save_path:       Path to save the plots
    :param use_pos:         If false, use particles only for shape parameters and fuse position in Kalman fashion
    """
    # store prior for plotting
    m_prior = mmsr_pf['x'][M]
    l_prior, w_prior, al_prior = get_ellipse_params_from_sr(mmsr_pf['x'][SR])

    # use particle filter
    mmsr_pf['x'], mmsr_pf['weights'], mmsr_pf['cov'][:2, :2] = particle_filter(particles_pf, mmsr_pf['weights'], meas,
                                                                               cov_meas, n_particles_pf, 'sum',
                                                                               mmsr_pf['x'][M], mmsr_pf['cov'][:2, :2],
                                                                               use_pos)

    # save error and plot estimate
    l_post, w_post, al_post = get_ellipse_params_from_sr(mmsr_pf['x'][SR])
    mmsr_pf['error'][i::steps] += error_and_plotting(mmsr_pf['x'][M], l_post, w_post, al_post, m_prior, l_prior,
                                                     w_prior, al_prior, meas[M], meas[L], meas[W], meas[AL], gt[M],
                                                     gt[L], gt[W], gt[AL], plot_cond, 'MMGW-PF',
                                                     save_path + 'exampleMMGWPF%i.svg' % i, est_color='green')
