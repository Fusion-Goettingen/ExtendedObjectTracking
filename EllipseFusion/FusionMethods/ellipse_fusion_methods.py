"""
Author: Kolja Thormann

Contains various ellipse fusion methods
"""

import numpy as np

from FusionMethods.ellipse_fusion_support import particle_filter, mwdp_fusion, get_jacobian, get_ellipse_params,\
    get_ellipse_params_from_sr, single_particle_approx_gaussian, to_matrix, turn_to_multi_modal
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


def mmsr_lin2_update(mmsr_lin2, meas, cov_meas, gt, i, steps, plot_cond, save_path):
    """
    Fuse using MMSR-Lin2; store state in square root space and estimate measurement in square root space by transforming
    the measurement covariance using Hessians of the transformation function; Hessian formulas based on M. Roth and
    F. Gustafsson, “An Efficient Implementation of the Second Order Extended Kalman Filter,” in Proceedings of the 14th
    International Conference  on  Information  Fusion  (Fusion  2011),  Chicago,  Illinois, USA, July 2011.
    :param mmsr_lin2:   Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param gt:          Ground truth
    :param i:           Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    """
    # convert measurement
    shape_meas_sr = to_matrix(meas[AL], meas[L], meas[W], True)

    # store prior for plotting
    m_prior = mmsr_lin2['x'][M]
    l_prior, w_prior, al_prior = get_ellipse_params_from_sr(mmsr_lin2['x'][SR])

    # precalculate values
    cossin = np.cos(meas[AL]) * np.sin(meas[AL])
    cos2 = np.cos(meas[AL]) ** 2
    sin2 = np.sin(meas[AL]) ** 2

    # transform per element
    meas_lin2 = np.zeros(5)
    meas_lin2[M] = meas[M]
    hess = np.zeros((3, 5, 5))
    hess[0] = np.array([
        [0, 0, 0,                               0,                0],
        [0, 0, 0,                               0,                0],
        [0, 0, 2*(meas[W]-meas[L])*(cos2-sin2), -2*cossin, 2*cossin],
        [0, 0, -2*cossin,                       0,                0],
        [0, 0, 2*cossin,                        0,                0],
    ])
    meas_lin2[2] = shape_meas_sr[0, 0] + 0.5 * np.trace(np.dot(hess[0], cov_meas))
    hess[1] = np.array([
        [0, 0, 0,                           0,                 0],
        [0, 0, 0,                           0,                 0],
        [0, 0, -4*(meas[W]-meas[L])*cossin, cos2-sin2, sin2-cos2],
        [0, 0, cos2-sin2,                   0,                 0],
        [0, 0, sin2-cos2,                   0,                 0],
    ])
    meas_lin2[3] = shape_meas_sr[0, 1] + 0.5 * np.trace(np.dot(hess[1], cov_meas))
    hess[2] = np.array([
        [0, 0, 0,                               0,                0],
        [0, 0, 0,                               0,                0],
        [0, 0, 2*(meas[L]-meas[W])*(cos2-sin2), 2*cossin, -2*cossin],
        [0, 0, 2*cossin,                        0,                0],
        [0, 0, -2*cossin,                       0,                0],
    ])
    meas_lin2[4] = shape_meas_sr[1, 1] + 0.5 * np.trace(np.dot(hess[2], cov_meas))

    # transform covariance per element
    jac = get_jacobian(meas[L], meas[W], meas[AL])
    cov_meas_lin2 = np.dot(np.dot(jac, cov_meas), jac.T)
    # add Hessian part where Hessian not 0
    for k in range(3):
        for l in range(3):
            cov_meas_lin2[k+2, l+2] += 0.5 * np.trace(np.dot(np.dot(np.dot(hess[k], cov_meas), hess[l]), cov_meas))

    # Kalman fusion
    S_lin = mmsr_lin2['cov'] + cov_meas_lin2
    S_lin_inv = np.linalg.inv(S_lin)
    if np.iscomplex(S_lin_inv).any():
        print(cov_meas_lin2)
        print(S_lin_inv)
    K_lin = np.dot(mmsr_lin2['cov'], S_lin_inv)
    mmsr_lin2['x'] = mmsr_lin2['x'] + np.dot(K_lin, meas_lin2 - mmsr_lin2['x'])
    mmsr_lin2['cov'] = mmsr_lin2['cov'] - np.dot(np.dot(K_lin, S_lin), K_lin.T)

    # save error and plot estimate
    l_post, w_post, al_post = get_ellipse_params_from_sr(mmsr_lin2['x'][SR])
    mmsr_lin2['error'][i::steps] += error_and_plotting(mmsr_lin2['x'][M], l_post, w_post, al_post, m_prior, l_prior,
                                                       w_prior, al_prior, meas[M], meas[L], meas[W], meas[AL], gt[M],
                                                       gt[L], gt[W], gt[AL], plot_cond, 'Linearization',
                                                       save_path + 'exampleLin%i.svg' % i)


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


def multi_modal_update(mmsr_mult_prior, mmsr_mult_prior_cov, meas, cov_meas, n_particles, prior, gt, plot_cond,
                       save_path):
    """
    Fuses the multi modal density, with each mode representing a different way to parameterize the same ellipse, with a
    measurement, creating a 16 component density. Next, the density is transformed and averaged in square root space via
    sampling of particles.
    :param mmsr_mult_prior:     4 component prior means
    :param mmsr_mult_prior_cov: Covariances of the components
    :param meas:                Measurement in original state space
    :param cov_meas:            Covariance of measurement in original state space
    :param n_particles:         Number of particles used for approximating the transformed density
    :param prior:               Prior estimate (for plotting
    :param gt:                  Ground truth
    :param plot_cond:           Boolean determining whether to plot the current estimate
    :param save_path:           Path to save the plots
    :return:                    The GW and SR error of the fusion
    """
    mmsr_mult_meas, mmsr_mult_meas_cov = turn_to_multi_modal(meas, cov_meas)
    mmsr_mult_post = np.zeros((16, 5))
    mmsr_mult_post_cov = np.zeros((16, 5, 5))
    mmsr_mult_post_weights = np.zeros(16)
    for i in range(4):
        for j in range(4):
            nu = mmsr_mult_meas[j] - mmsr_mult_prior[i]
            nu[2] = (nu[2] + np.pi) % (2*np.pi) - np.pi
            nu_cov = mmsr_mult_prior_cov[i] + mmsr_mult_meas_cov[j]
            mmsr_mult_post[i*4+j] = mmsr_mult_prior[i] + np.dot(np.dot(mmsr_mult_prior_cov[i],
                                                                       np.linalg.inv(nu_cov)), nu)
            mmsr_mult_post_cov[i*4+j] = mmsr_mult_prior_cov[i] - np.dot(np.dot(mmsr_mult_prior_cov[i],
                                                                               np.linalg.inv(nu_cov)),
                                                                        mmsr_mult_prior_cov[i].T)
            mmsr_mult_post_weights[i*4+j] = -2.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(nu_cov)) \
                                            - 0.5*np.dot(np.dot(nu, np.linalg.inv(nu_cov)), nu)

    mmsr_mult_post_weights -= np.log(np.sum(np.exp(mmsr_mult_post_weights)))
    mmsr_mult_post_weights = np.exp(mmsr_mult_post_weights)

    # sample from multimodal density to approximate transformation
    chosen = np.random.choice(16, n_particles, True, p=mmsr_mult_post_weights)
    particle = np.zeros((n_particles, 5))
    for i in range(16):
        if np.sum(chosen == i) > 0:
            particle[chosen == i] = np.random.multivariate_normal(mmsr_mult_post[i], mmsr_mult_post_cov[i],
                                                                  np.sum(chosen == i))
            # transform
            mult_sr = to_matrix(particle[chosen == i, AL], particle[chosen == i, L], particle[chosen == i, W], True)
            particle[chosen == i, 2] = mult_sr[:, 0, 0]
            particle[chosen == i, 3] = mult_sr[:, 0, 1]
            particle[chosen == i, 4] = mult_sr[:, 1, 1]

    # calculate mean
    mmsr_mult_final = np.mean(particle, axis=0)
    mmsr_mult_final_l, mmsr_mult_final_w, mmsr_mult_final_al = get_ellipse_params_from_sr(mmsr_mult_final[2:])

    return error_and_plotting(mmsr_mult_final[M], mmsr_mult_final_l, mmsr_mult_final_w, mmsr_mult_final_al, prior[M],
                              prior[L], prior[W], prior[AL], meas[M], meas[L], meas[W], meas[AL], gt[M], gt[L], gt[W],
                              gt[AL], plot_cond, 'Multimodal', save_path + 'exampleMCApprox%i.svg' % 0,
                              est_color='green')
