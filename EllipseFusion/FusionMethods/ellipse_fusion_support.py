"""
Author: Kolja Thormann

Contains support functions for the ellipse fusion and test setup
"""

import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.linalg import inv, sqrtm
from numpy.linalg import slogdet

from FusionMethods.constants import *


def rot_matrix(alpha):
    """
    Calculates a rotation matrix based on the input orientation
    :param alpha:   Input orientation
    :return:        Rotation matrix for alpha
    """
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    if len(rot.shape) == 3:
        return rot.transpose((2, 0, 1))
    else:
        return rot


def to_matrix(alpha, l, w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix depending on sr parameter
    :param alpha:   Orientation of the ellipse
    :param l:       Semi-axis length of the ellipse
    :param w:       Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix depending of sr
    """
    p = 1 if sr else 2
    rot = rot_matrix(alpha)
    if len(rot.shape) == 3:
        lw_diag = np.array([np.diag([l[i], w[i]]) for i in range(len(l))])
        return np.einsum('xab, xbc, xdc -> xad', rot, lw_diag ** p, rot)
    else:
        return np.dot(np.dot(rot, np.diag([l, w]) ** p), rot.T)


def get_ellipse_params(ell):
    """
    Calculate the ellipse semi-axis length and width and orientation based on shape matrix
    :param ell: Input ellipse as 2x2 shape matrix
    :return:    Semi-axis length, width and orientation of input ellipse
    """
    ellipse_axis, v = np.linalg.eig(ell)
    ellipse_axis = np.sqrt(ellipse_axis)
    l = ellipse_axis[0]
    w = ellipse_axis[1]
    al = np.arctan2(v[1, 0], v[0, 0])

    return l, w, al


def get_ellipse_params_from_sr(sr):
    """
    Calculate ellipse semi-axis length and width and orientation based on the elements of the square root matrix
    :param sr:  Elements of the square root matrix [top-left, corner, bottom-right]
    :return:    Semi-axis length, width and orientation of input square root matrix
    """
    # calculate shape matrix based on square root matrix elements
    ell_sr = np.array([
        [sr[0], sr[1]],
        [sr[1], sr[2]],
    ])
    ell = np.dot(ell_sr, ell_sr)

    return get_ellipse_params(ell)


def single_particle_approx_gaussian(prior, cov, n_particles, use_pos, sr=True):
    """
    Calculate the particle density of the prior in square root space and approximate it as a Gaussian
    :param prior:       Prior in original state space
    :param cov:         Covariance of the prior
    :param n_particles: Number of particles used for particle approximation
    :param use_pos:     Boolean to determine whether to create the particle cloud for the entire or only the shape state
    :param sr:          Utilize square root matrix or normal matrix for particles
    :return:            Approximated mean and covariance in square root space and the particle density
    """
    vec_sr = np.zeros((n_particles, 5))

    particle = sample_m(prior, cov, False, n_particles)  # sample particles from the prior density
    if not use_pos:
        particle[:, M] = prior[M]

    # transform particles into square root space
    for i in range(n_particles):
        # calculate square root
        Pa_sr = to_matrix(particle[i, AL], particle[i, L], particle[i, W], sr)

        # save transformed particle
        vec_sr[i] = np.array([particle[i, 0], particle[i, 1], Pa_sr[0, 0], Pa_sr[0, 1], Pa_sr[1, 1]])

    # calculate mean and variance of particle density
    if use_pos:
        mean_sr = np.sum(vec_sr, axis=0) / n_particles
        var_sr = np.sum(np.einsum('xa, xb -> xab', vec_sr - mean_sr, vec_sr - mean_sr), axis=0) / n_particles
    else:
        mean_sr = np.zeros(5)
        var_sr = np.zeros((5, 5))
        mean_sr[M] = prior[M]
        var_sr[:2, :2] = cov[:2, :2]
        mean_sr[SR] = np.sum(vec_sr[:, SR], axis=0) / n_particles
        var_sr[2:, 2:] = np.sum(np.einsum('xa, xb -> xab', vec_sr[:, SR] - mean_sr[SR], vec_sr[:, SR] - mean_sr[SR]),
                                axis=0) / n_particles
    var_sr += var_sr.T
    var_sr *= 0.5

    return mean_sr, var_sr, vec_sr


def sample_m(mean, cov, shift, amount):
    """
    Create one or multiple samples
    :param mean:    Mean from which to sample
    :param cov:     Covariance with which to sample
    :param shift:   Boolean to determine whether to shift the orientation
    :param amount:  Number of samples to be drawn
    :return:        The sample for amount==1 and an array of samples for amount > 1
    """
    # shift mean by 90 degree and switch length and width if demanded
    s_mean = mean.copy()
    if shift:
        save_w = s_mean[W]
        s_mean[W] = s_mean[L]
        s_mean[L] = save_w
        s_mean[AL] += 0.5 * np.pi

    # draw sample
    samp = mvn(s_mean, cov, amount)

    # enforce restrictions
    samp[:, L] = np.maximum(0.1, samp[:, L])
    samp[:, W] = np.maximum(0.1, samp[:, W])
    samp[:, AL] %= 2 * np.pi

    # if only one sample, do not store it in a 1d array
    if amount == 1:
        samp = samp[0]

    return samp


def particle_filter(prior, w, meas, cov_meas, n_particles, ll_type, m_prior, cov_m, use_pos):
    """
    Update the particle cloud's (in SR space) weights based on a measurement in original state space; no resampling
    :param prior:       Prior particle density in SR space (shape space if use_bary is True)
    :param w:           Prior weights
    :param meas:        Measured ellipse in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param n_particles: Number of particles to be sampled from the Gaussian input density
    :param ll_type:     Either 'sum' for using the sum of the 4 representations' likelihood or 'max' for using the
                        maximum
    :param m_prior:     Prior of position parameters (only used if use_pos is false)
    :param cov_m:       Covariance of position parameters (only used if use_pos is false)
    :param use_pos:     If false, use particles only for shape parameters and fuse position in Kalman fashion
    :return:            Weighted mean of the particles and updated weights
    """

    meas_mm, cov_meas_mm = turn_to_multi_modal(meas, cov_meas)
    # calculate inverse and determinant of measurement covariance assuming independent measurement dimensions
    # if use_pos:
    #     cov_meas_inv = np.diag(1.0 / cov_meas.diagonal())
    #     cov_meas_det = np.linalg.det(cov_meas)
    # else:
    #     cov_meas_inv = np.diag(1.0 / cov_meas[2:, 2:].diagonal())
    #     cov_meas_det = np.linalg.det(cov_meas[2:, 2:])
    if use_pos:
        cov_meas_inv = np.zeros(cov_meas_mm.shape)
        cov_meas_det = np.zeros(len(cov_meas_mm))
    else:
        cov_meas_inv = np.zeros(cov_meas_mm[:, 2:, 2:].shape)
        cov_meas_det = np.zeros(len(cov_meas_mm))
    for i in range(4):
        if use_pos:
            cov_meas_inv[i] = np.diag(1.0 / cov_meas_mm[i].diagonal())
            cov_meas_det[i] = np.linalg.det(cov_meas_mm[i])
        else:
            cov_meas_inv[i] = np.diag(1.0 / cov_meas_mm[i, 2:, 2:].diagonal())
            cov_meas_det[i] = np.linalg.det(cov_meas_mm[i, 2:, 2:])

    # update weights with likelihood
    for i in range(n_particles):
        w[i] *= sr_likelihood(prior[i], cov_meas_inv[0], cov_meas_det[0], meas_mm[0], ll_type)
    w /= np.sum(w)

    # calculate weighted mean with updated particle weights
    post_mean = np.sum(w[:, None] * prior, axis=0)

    if not use_pos:
        S = cov_meas[:2, :2] + cov_m
        K = np.dot(cov_m, inv(S))
        post_mean[M] = m_prior + np.dot(K, meas[M] - m_prior)
        cov_m = cov_m - np.dot(np.dot(K, S), K.T)

    return post_mean, w, cov_m


def sr_likelihood(x, cov_inv, cov_det, meas, ll_type):
    """
    Calculate the likelihood of a particle in SR space given a measurement in original state space
    :param x:       The particle in SR space
    :param cov_inv: Inverse of measurement covariance
    :param cov_det: Determinant of measurement covariance
    :param meas:    Measurement in original state space
    :param ll_type: Either 'sum' for using the sum of the 4 representations' likelihood or 'max' for using the maximum
    :return:        Depedning on ll_type the sum or maximum of the likelihoods or 0 for an invalid ll_type
    """
    # transform particle to original state space
    x_el = np.zeros(5)
    x_el[:2] = x[:2]
    x_shape = np.array([
        [x[2], x[3]],
        [x[3], x[4]],
    ])
    x_shape = np.dot(x_shape, x_shape)
    l, w, al = get_ellipse_params(x_shape)
    x_el[2] = al - 0.5*np.pi
    x_el[2] %= 2.0*np.pi
    x_el[3] = w
    x_el[4] = l

    ll = np.zeros(4)

    # calculate likelihood for all 4 representations in original state space
    for i in range(4):
        save = x_el[3]
        x_el[3] = x_el[4]
        x_el[4] = save
        x_el[2] += 0.5*np.pi
        x_el[2] %= 2.0*np.pi

        nu = meas - x_el
        nu[2] = ((nu[2] + np.pi) % (2*np.pi)) - np.pi

        if len(cov_inv) == 5:
            ll[i] = np.exp(-0.5*np.dot(np.dot(nu, cov_inv), nu)) / np.sqrt(32*np.pi**5*cov_det)
        else:
            ll[i] = np.exp(-0.5 * np.dot(np.dot(nu[2:], cov_inv), nu[2:])) / np.sqrt(8 * np.pi ** 3 * cov_det)

    # return sum or maximum depending on ll_type
    if ll_type == 'sum':
        return np.sum(ll)
    elif ll_type == 'max':
        return np.max(ll)
    else:
        print('Invalid likelihood type')
        return 0


def mwdp_fusion(prior, cov_prior, meas, cov_meas):
    """
    Fuse ellipses A and B in original state space using representation with highest likelihood; assumes independent
    measurement dimensions for the switch of covariance elements
    :param prior:       Prior ellipse in original state space
    :param cov_prior:   Covariance of prior ellipse
    :param meas:        Measured ellipse in original state space
    :param cov_meas:    Covariance of measured ellipse
    :return:            Mean and covariance of fusion with highest likelihood representation and number of 90 degree
                        shifts of ellipse B to get highest likelihood representation
    """
    res_orig_alt_rots = np.zeros((4, 5))
    res_orig_alt_rots_cov = np.zeros((4, 5, 5))
    res_orig_log_lik = np.zeros(4)
    innov = np.zeros((4, 5))
    meas_alt = np.zeros(5)
    meas_alt[M] = meas[M]

    # test all 4 representations
    for k in range(4):
        # shift orientation and if necessary switch semi-axis in mean and orientation
        meas_alt[AL] = (meas[AL] + k * np.pi * 0.5) % (2 * np.pi)
        if k % 2 != 0:
            meas_alt[L] = meas[W]
            meas_alt[W] = meas[L]
            cov_meas_alt = np.copy(cov_meas)
            cov_meas_alt[3, 3] = cov_meas[4, 4]
            cov_meas_alt[4, 4] = cov_meas[3, 3]
        else:
            meas_alt[L] = meas[L]
            meas_alt[W] = meas[W]
            cov_meas_alt = np.copy(cov_meas)

        # Kalman update
        S_orig_alt = cov_prior + cov_meas_alt
        K_orig_alt = np.dot(cov_prior, np.linalg.inv(S_orig_alt))
        innov[k] = meas_alt - prior
        # use shorter angle difference
        innov[k, 2] = ((innov[k, 2] + np.pi) % (2*np.pi)) - np.pi
        res_orig_alt_rots[k] = prior + np.dot(K_orig_alt, innov[k])
        res_orig_alt_rots_cov[k] = cov_prior - np.dot(np.dot(K_orig_alt, S_orig_alt), K_orig_alt.T)

        # calculate log-likelihood
        res_orig_log_lik[k] = -0.5 * np.dot(np.dot(innov[k], inv(S_orig_alt)), innov[k])
        sign, logdet_inv = slogdet(inv(S_orig_alt))
        res_orig_log_lik[k] += 0.5 * logdet_inv - 2.5 * np.log(2 * np.pi)

    return res_orig_alt_rots[np.argmax(res_orig_log_lik)], res_orig_alt_rots_cov[np.argmax(res_orig_log_lik)],\
           np.argmax(res_orig_log_lik)


def get_jacobian(l, w, al):
    """
    Jacobian calculation of square root function for positive semi-definite matrices with eigenvalues l and w and
    rotation matrix with orientation al
    :param l:   Semi-axis length of original representation
    :param w:   Semi-axis width of original representation
    :param al:  Orientation of original representation
    :return:    Jacobian of square root transformation
    """
    jac = np.zeros((5, 5))
    jac[:2, :2] = np.eye(2)

    jac[2, 2] = 2 * np.cos(al) * np.sin(al) * (w - l)
    jac[2, 3] = np.cos(al)**2
    jac[2, 4] = np.sin(al)**2

    jac[3, 2] = (l - w) * (np.cos(al)**2 - np.sin(al)**2)
    jac[3, 3] = np.sin(al) * np.cos(al)
    jac[3, 4] = -np.sin(al) * np.cos(al)

    jac[4, 2] = 2 * np.cos(al) * np.sin(al) * (l - w)
    jac[4, 3] = np.sin(al)**2
    jac[4, 4] = np.cos(al)**2

    return jac


def barycenter(particles, w, n_particles, particles_sr=np.zeros(0)):
    """
    Determine Barycenter of particles in shape space via optimization (based on G. Puccetti, L. Rüschendorf, and
    S. Vanduffel, “On the Computation of Wasserstein Barycenters,” Available at SSRN 3276147, 2018).
    :param particles:       Particles in shape space [m1, m2, c11, c12, c22] with cnm being members of the covariance
                            matrix
    :param w:               Weights of the particles
    :param n_particles:     Number of particles
    :param particles_sr:    Particles with shape parameters in SR form; if given, used for initial guess
    :return:                Barycenter of the particles as [m1, m2, c11, c12, c22]
    """
    # Calculate covariances
    covs = np.zeros((n_particles, 2, 2))
    covs[:, 0, 0] = particles[:, 2]
    covs[:, 0, 1] = particles[:, 3]
    covs[:, 1, 0] = particles[:, 3]
    covs[:, 1, 1] = particles[:, 4]

    # Calculate Barycenter
    if len(particles_sr) > 0:
        covs_sr = np.zeros((n_particles, 2, 2))
        covs_sr[:, 0, 0] = particles_sr[:, 2]
        covs_sr[:, 0, 1] = particles_sr[:, 3]
        covs_sr[:, 1, 0] = particles_sr[:, 3]
        covs_sr[:, 1, 1] = particles_sr[:, 4]
        bary_sr = np.sum(w[:, None, None] * covs_sr, axis=0)
        bary = np.dot(bary_sr, bary_sr)
    else:
        bary = np.eye(2)
        bary_sr = np.eye(2)
    conv = False
    # loop until convergence
    while not conv:
        res = np.zeros((n_particles, 2, 2))
        for i in range(n_particles):
            res[i] = np.dot(np.dot(bary_sr, covs[i]), bary_sr)
            res[i] = sqrtm(res[i]) * w[i]
        bary_new = np.sum(res, axis=0)
        bary_sr = sqrtm(bary_new)

        # check convergence
        # bary_new = np.dot(bary_sr, bary_sr)
        diff = np.sum(abs(bary_new - bary))
        conv = diff < 1e-6
        bary = bary_new

    # Calculate mean and Barycenter in SR space
    result = np.zeros(5)
    result[M] = np.sum(w[:, None] * particles[:, M], axis=0)
    result[2] = bary[0, 0]
    result[3] = bary[0, 1]
    result[4] = bary[1, 1]

    return result


def turn_to_multi_modal(mean, cov):
    """
    Turns an ellipse estimate in original state space into a multi modal density consisting of 4 components, one for
    each way of representing the same ellipse (reducing the number of possibilities to 4 utilizing the 2pi periodity of
    the orientation).
    :param mean:    Mean of the density
    :param cov:     Covariance of the density
    :return:        Means and covariances of all 4 modes
    """
    mmsr_mult_mean = np.zeros((4, 5))
    mmsr_mult_cov = np.zeros((4, 5, 5))
    mmsr_mult_mean[0] = mean.copy()
    mmsr_mult_cov[0] = cov.copy()
    for i in range(1, 4):
        mmsr_mult_mean[i, :2] = mmsr_mult_mean[i - 1, :2]
        mmsr_mult_mean[i, 2] = (mmsr_mult_mean[i - 1, 2] + 0.5 * np.pi) % (2 * np.pi)
        mmsr_mult_mean[i, 3] = mmsr_mult_mean[i - 1, 4]
        mmsr_mult_mean[i, 4] = mmsr_mult_mean[i - 1, 3]
        mmsr_mult_cov[i, :3, :3] = mmsr_mult_cov[i - 1, :3, :3]
        mmsr_mult_cov[i, 3, 3] = mmsr_mult_cov[i - 1, 4, 4]
        mmsr_mult_cov[i, 4, 4] = mmsr_mult_cov[i - 1, 3, 3]

    return mmsr_mult_mean, mmsr_mult_cov
