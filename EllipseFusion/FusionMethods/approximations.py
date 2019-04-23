"""
Author: Kolja Thormann
"""

import numpy as np
from numpy.random import normal
from numpy.random import multivariate_normal as mvn
from scipy.linalg import sqrtm, inv
from numpy.linalg import slogdet

from FusionMethods.helpers import *


def particle_approx(n_particles, m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                    plotting):
    l_pa = np.zeros(n_particles)
    w_pa = np.zeros(n_particles)
    al_pa = np.zeros(n_particles)
    vec_A = np.zeros((n_particles, 5))
    ma_final = 0
    Pa_final = np.zeros((2, 2))

    l_pb = np.zeros(n_particles)
    w_pb = np.zeros(n_particles)
    al_pb = np.zeros(n_particles)
    vec_B = np.zeros((n_particles, 5))
    mb_final = 0
    Pb_final = np.zeros((2, 2))

    i = 0
    while i < n_particles:
        # if i % 100 == 0:
        #     print(i)

        m_pa = mvn(m_a, cov_a[:2, :2])

        l_pa[i] = np.maximum(normal(l_a, cov_a[3, 3]), 0.1)
        w_pa[i] = np.maximum(normal(w_a, cov_a[4, 4]), 0.1)
        al_pa[i] = normal(al_a, cov_a[2, 2])
        al_pa[i] %= (2 * np.pi)
        rot_p = np.array([
            [np.cos(al_pa[i]), -np.sin(al_pa[i])],
            [np.sin(al_pa[i]), np.cos(al_pa[i])]
        ])

        Pa = np.dot(np.dot(rot_p, np.diag([l_pa[i], w_pa[i]]) ** 2), rot_p.T)

        Pa_sr = sqrtm(Pa)

        ma_final += m_pa

        Pa_final += Pa_sr

        vec_A[i] = np.array([m_pa[0], m_pa[1], Pa_sr[0, 0], Pa_sr[0, 1], Pa_sr[1, 1]])

        m_pb = mvn(m_b, cov_b[:2, :2])

        # switch l and w and rotate 90 degree
        l_pb[i] = np.maximum(normal(l_b, cov_b[3, 3]), 0.1)
        w_pb[i] = np.maximum(normal(w_b, cov_b[4, 4]), 0.1)
        al_pb[i] = normal(al_b, cov_b[2, 2])
        al_pb[i] %= (2 * np.pi)
        rot_p = np.array([
            [np.cos(al_pb[i]), -np.sin(al_pb[i])],
            [np.sin(al_pb[i]), np.cos(al_pb[i])]
        ])

        Pb = np.dot(np.dot(rot_p, np.diag([l_pb[i], w_pb[i]]) ** 2), rot_p.T)

        Pb_sr = sqrtm(Pb)

        mb_final += m_pb

        Pb_final += Pb_sr

        vec_B[i] = np.array([m_pb[0], m_pb[1], Pb_sr[0, 0], Pb_sr[0, 1], Pb_sr[1, 1]])

        i += 1

    ma_final /= n_particles
    Pa_final /= n_particles
    mean_A = np.sum(vec_A, axis=0) / n_particles
    var_A = np.sum(np.einsum('xa, xb -> xab', vec_A - mean_A, vec_A - mean_A), axis=0) / n_particles
    var_A += var_A.T
    var_A *= 0.5

    mb_final /= n_particles
    Pb_final /= n_particles
    mean_B = np.sum(vec_B, axis=0) / n_particles
    var_B = np.sum(np.einsum('xa, xb -> xab', vec_B - mean_B, vec_B - mean_B), axis=0) / n_particles
    var_B += var_B.T
    var_B *= 0.5

    # fuse A and B via mean and variance
    S = var_A + var_B
    K = np.dot(var_A, np.linalg.inv(S))
    mean_Res = mean_A + np.dot(K, mean_B - mean_A)
    # var_Res = var_A - np.dot(np.dot(K, S), K.T)
    m_res = mean_Res[:2]
    Res_sr = np.array([
        [mean_Res[2], mean_Res[3]],
        [mean_Res[3], mean_Res[4]],
    ])
    Res = np.dot(Res_sr, Res_sr)

    # turn result back and compare
    ellipse_axisres, vres = np.linalg.eig(Res)
    ellipse_axisres = np.sqrt(ellipse_axisres)
    l_res = ellipse_axisres[0]
    w_res = ellipse_axisres[1]
    al_res = np.arctan2(vres[1, 0], vres[0, 0])

    if plotting:
        plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, m_res, l_res, w_res, al_res,
                      'MC Approximated Fusion', 'exampleMCApprox.svg')

    return gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, m_res, l_res, w_res, al_res)


def ord_fusion(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, plotting):
    S_orig = cov_a + cov_b
    if np.linalg.det(S_orig) == 0:
        print('Singular S_orig')
        # print(S_orig)

    K_orig = np.dot(cov_a, np.linalg.inv(S_orig))
    res_orig = np.array([m_a[0], m_a[1], al_a, l_a, w_a]) + np.dot(K_orig, np.array([m_b[0], m_b[1], al_b, l_b, w_b])
                                                                   - np.array([m_a[0], m_a[1], al_a, l_a, w_a]))

    if plotting:
        plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, res_orig[:2], res_orig[3],
                      res_orig[4], res_orig[2], 'Fusion of Original State', 'exampleRegFus.svg')

    return gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, res_orig[:2], res_orig[3], res_orig[4], res_orig[2])


def best_fusion(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b):
    res_orig_alt_rots = np.zeros((4, 5))
    res_orig_log_lik = np.zeros(4)
    innov = np.zeros((4, 5))
    for k in range(4):
        al_b_alt = (al_b + k * np.pi * 0.5) % (2 * np.pi)
        if k % 2 != 0:
            l_b_alt = w_b
            w_b_alt = l_b
            cov_b_alt = np.copy(cov_b)
            cov_b_alt[3, 3] = cov_b[4, 4]
            cov_b_alt[4, 4] = cov_b[3, 3]
        else:
            l_b_alt = l_b
            w_b_alt = w_b
            cov_b_alt = np.copy(cov_b)
        S_orig_alt = cov_a + cov_b_alt
        if np.linalg.det(S_orig_alt) == 0:
            print('Singular S_orig')
            # print(S_orig)
            continue
        K_orig_alt = np.dot(cov_a, np.linalg.inv(S_orig_alt))
        innov[k] = np.array([m_b[0], m_b[1], al_b_alt, l_b_alt, w_b_alt]) - np.array([m_a[0], m_a[1], al_a, l_a, w_a])
        # use shorter angle difference
        innov[k, 2] = ((innov[k, 2] + np.pi) % (2*np.pi)) - np.pi
        res_orig_alt_rots[k] = np.array([m_a[0], m_a[1], al_a, l_a, w_a]) + np.dot(K_orig_alt, innov[k])
        res_orig_log_lik[k] = -0.5 * np.dot(np.dot(innov[k], inv(S_orig_alt)), innov[k])
        sign, logdet_inv = slogdet(inv(S_orig_alt))
        res_orig_log_lik[k] += 0.5 * logdet_inv - 0.5 * 5 * np.log(2 * np.pi)

    return res_orig_alt_rots[np.argmax(res_orig_log_lik)], np.argmax(res_orig_log_lik)


def best_or(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, plotting):
    [res_orig_alt, k] = best_fusion(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b)

    if plotting:
        plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, res_orig_alt[:2],
                      res_orig_alt[3], res_orig_alt[4], res_orig_alt[2],
                      'Linearized Approximated Fusion With Best Orientation', 'exampleRegAltFus.svg')

    return gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, res_orig_alt[:2], res_orig_alt[3], res_orig_alt[4],
                             res_orig_alt[2])


def shape_mean(m_a, A, l_a, w_a, al_a, m_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, plotting):
    m_res_alt2 = 0.5 * (m_a + m_b)
    Res_alt2 = 0.5 * (A + B)
    ellipse_axisalt2, valt2 = np.linalg.eig(Res_alt2)
    ellipse_axisalt2 = np.sqrt(ellipse_axisalt2)
    l_res_alt2 = ellipse_axisalt2[0]
    w_res_alt2 = ellipse_axisalt2[1]
    al_res_alt2 = np.arctan2(valt2[1, 0], valt2[0, 0])

    if plotting:
        plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, m_res_alt2, l_res_alt2,
                      w_res_alt2, al_res_alt2, 'Shape Mean', 'exampleMean.svg')

    return gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, m_res_alt2, l_res_alt2, w_res_alt2, al_res_alt2)


def lin_approx(m_a, cov_a, A, l_a, w_a, al_a, m_b, cov_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, plotting):
    jac_a = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, (w_a ** 2 - l_a ** 2) * np.sin(2 * al_a), 2 * l_a * np.cos(al_a) ** 2, 2 * w_a * np.sin(al_a) ** 2],
        [0, 0, (l_a ** 2 - w_a ** 2) * np.cos(2 * al_a), 2 * l_a * np.cos(al_a) * np.sin(al_a),
         -2 * w_a * np.sin(al_a) * np.cos(al_a)],
        [0, 0, (l_a ** 2 - w_a ** 2) * np.sin(2 * al_a), 2 * l_a * np.sin(al_a) ** 2, 2 * w_a * np.cos(al_a) ** 2],
    ])
    jac_b = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, (w_b ** 2 - l_b ** 2) * np.sin(2 * al_b), 2 * l_b * np.cos(al_b) ** 2, 2 * w_b * np.sin(al_b) ** 2],
        [0, 0, (l_b ** 2 - w_b ** 2) * np.cos(2 * al_b), 2 * l_b * np.cos(al_b) * np.sin(al_b),
         -2 * w_b * np.sin(al_b) * np.cos(al_b)],
        [0, 0, (l_b ** 2 - w_b ** 2) * np.sin(2 * al_b), 2 * l_b * np.sin(al_b) ** 2, 2 * w_b * np.cos(al_b) ** 2],
    ])
    s_a = 2 * np.sqrt(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
    t_a = np.sqrt(A[0, 0] + A[1, 1] + s_a)
    s_a_al = (jac_a[2, 2] * A[1, 1] + jac_a[4, 2] * A[0, 0] - 2 * jac_a[3, 2] * A[1, 0]) / np.sqrt(
        A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
    t_a_al = (jac_a[2, 2] + jac_a[4, 2] + s_a_al) / (2 * t_a)
    s_a_l = (jac_a[2, 3] * A[1, 1] + jac_a[4, 3] * A[0, 0] - 2 * jac_a[3, 3] * A[1, 0]) / np.sqrt(
        A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
    t_a_l = (jac_a[2, 3] + jac_a[4, 3] + s_a_al) / (2 * t_a)
    s_a_w = (jac_a[2, 4] * A[1, 1] + jac_a[4, 4] * A[0, 0] - 2 * jac_a[3, 4] * A[1, 0]) / np.sqrt(
        A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])
    t_a_w = (jac_a[2, 4] + jac_a[4, 4] + s_a_al) / (2 * t_a)
    jac_a2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, -t_a_al * (A[0, 0] + s_a) / t_a ** 2 + (jac_a[2, 2] + s_a_al) / t_a,
         -t_a_l * (A[0, 0] + s_a) / t_a ** 2 + (jac_a[2, 3] + s_a_l) / t_a,
         -t_a_w * (A[0, 0] + s_a) / t_a ** 2 + (jac_a[2, 4] + s_a_w) / t_a],
        [0, 0, -t_a_al * A[1, 0] / t_a ** 2 + jac_a[3, 2] / t_a, -t_a_l * A[1, 0] / t_a ** 2 + jac_a[3, 3] / t_a,
         -t_a_w * A[1, 0] / t_a ** 2 + jac_a[3, 4] / t_a],
        [0, 0, -t_a_al * (A[1, 1] + s_a) / t_a ** 2 + (jac_a[4, 2] + s_a_al) / t_a,
         -t_a_l * (A[1, 1] + s_a) / t_a ** 2 + (jac_a[4, 3] + s_a_l) / t_a,
         -t_a_w * (A[1, 1] + s_a) / t_a ** 2 + (jac_a[4, 4] + s_a_w) / t_a],
    ])
    s_b = 2 * np.sqrt(B[0, 0] * B[1, 1] - B[1, 0] * B[0, 1])
    t_b = np.sqrt(B[0, 0] + B[1, 1] + s_b)
    s_b_al = (jac_b[2, 2] * B[1, 1] + jac_b[4, 2] * B[0, 0] - 2 * jac_b[3, 2] * B[1, 0]) / np.sqrt(
        B[0, 0] * B[1, 1] - B[1, 0] * B[0, 1])
    t_b_al = (jac_b[2, 2] + jac_b[4, 2] + s_b_al) / (2 * t_b)
    s_b_l = (jac_b[2, 3] * B[1, 1] + jac_b[4, 3] * B[0, 0] - 2 * jac_b[3, 3] * B[1, 0]) / np.sqrt(
        B[0, 0] * B[1, 1] - B[1, 0] * B[0, 1])
    t_b_l = (jac_b[2, 3] + jac_b[4, 3] + s_b_al) / (2 * t_b)
    s_b_w = (jac_b[2, 4] * B[1, 1] + jac_b[4, 4] * B[0, 0] - 2 * jac_b[3, 4] * B[1, 0]) / np.sqrt(
        B[0, 0] * B[1, 1] - B[1, 0] * B[0, 1])
    t_b_w = (jac_b[2, 4] + jac_b[4, 4] + s_b_al) / (2 * t_b)
    jac_b2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, -t_b_al * (B[0, 0] + s_b) / t_b ** 2 + (jac_b[2, 2] + s_b_al) / t_b,
         -t_b_l * (B[0, 0] + s_b) / t_b ** 2 + (jac_b[2, 3] + s_b_l) / t_b,
         -t_b_w * (B[0, 0] + s_b) / t_b ** 2 + (jac_b[2, 4] + s_b_w) / t_b],
        [0, 0, -t_b_al * B[1, 0] / t_b ** 2 + jac_b[3, 2] / t_b, -t_b_l * B[1, 0] / t_b ** 2 + jac_b[3, 3] / t_b,
         -t_b_w * B[1, 0] / t_b ** 2 + jac_b[3, 4] / t_b],
        [0, 0, -t_b_al * (B[1, 1] + s_b) / t_b ** 2 + (jac_b[4, 2] + s_b_al) / t_b,
         -t_b_l * (B[1, 1] + s_b) / t_b ** 2 + (jac_b[4, 3] + s_b_l) / t_b,
         -t_b_w * (B[1, 1] + s_b) / t_b ** 2 + (jac_b[4, 4] + s_b_w) / t_b],
    ])
    sqrt_A = sqrtm(A)
    sqrt_B = sqrtm(B)
    vec_A_direct = np.array([m_a[0], m_a[1], sqrt_A[0, 0], sqrt_A[0, 1], sqrt_A[1, 1]])
    vec_B_direct = np.array([m_b[0], m_b[1], sqrt_B[0, 0], sqrt_B[0, 1], sqrt_B[1, 1]])
    S_lin2 = np.dot(np.dot(jac_a2, cov_a), jac_a2.T) + np.dot(np.dot(jac_b2, cov_b), jac_b2.T)

    K2 = np.dot(np.dot(np.dot(jac_a2, cov_a), jac_a2.T), np.linalg.inv(S_lin2))
    res_lin_vec2 = vec_A_direct + np.dot(K2, vec_B_direct - vec_A_direct)
    res_lin_S2 = np.array([
        [res_lin_vec2[2], res_lin_vec2[3]],
        [res_lin_vec2[3], res_lin_vec2[4]],
    ])
    res_lin_S2 = np.dot(res_lin_S2, res_lin_S2)
    ellipse_axislin2, vlin2 = np.linalg.eig(res_lin_S2)
    ellipse_axislin2 = np.sqrt(ellipse_axislin2)
    l_res_lin2 = ellipse_axislin2[0]
    w_res_lin2 = ellipse_axislin2[1]
    al_res_lin2 = np.arctan2(vlin2[1, 0], vlin2[0, 0])

    if plotting:
        plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, res_lin_vec2[:2], l_res_lin2,
                      w_res_lin2, al_res_lin2, 'Linearized Approximated Fusion', 'exampleGWFus.svg')

    return gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, res_lin_vec2[:2], l_res_lin2, w_res_lin2, al_res_lin2)
