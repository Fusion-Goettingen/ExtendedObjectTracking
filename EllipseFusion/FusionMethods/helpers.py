"""
Author: Kolja Thormann
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as mvn
from numpy.linalg import norm
from scipy.linalg import sqrtm
from matplotlib.patches import Ellipse


def gauss_wasserstein(m_x, l_x, w_x, al_x, m_y, l_y, w_y, al_y):
    gt_xc = m_x
    gt_alpha = al_x
    gt_eigen_val = np.array([l_x, w_x])
    gt_eigen_vec = np.array([
        [np.cos(gt_alpha), -np.sin(gt_alpha)],
        [np.sin(gt_alpha), np.cos(gt_alpha)],
    ])
    gt_sigma = np.einsum('ab, bc, dc -> ad', gt_eigen_vec, np.diag(gt_eigen_val ** 2), gt_eigen_vec)
    gt_sigma += gt_sigma.T
    gt_sigma /= 2

    track_xc = m_y
    track_alpha = al_y
    track_eigen_val = np.array([l_y, w_y])
    track_eigen_vec = np.array([
        [np.cos(track_alpha), -np.sin(track_alpha)],
        [np.sin(track_alpha), np.cos(track_alpha)],
    ])
    track_sigma = np.einsum('ab, bc, dc -> ad', track_eigen_vec, np.diag(track_eigen_val ** 2), track_eigen_vec)
    track_sigma += track_sigma.T
    track_sigma /= 2

    error = norm(gt_xc - track_xc) ** 2 + np.trace(gt_sigma + track_sigma
                                                   - 2 * sqrtm(np.einsum('ab, bc, cd -> ad',
                                                                         sqrtm(gt_sigma), track_sigma,
                                                                         sqrtm(gt_sigma))))
    error = np.sqrt(error)
    return error


def square_root_distance(m_x, l_x, w_x, al_x, m_y, l_y, w_y, al_y):
    x_rot = np.array([
        [np.cos(al_x), -np.sin(al_x)],
        [np.sin(al_x),  np.cos(al_x)],
    ])
    y_rot = np.array([
        [np.cos(al_y), -np.sin(al_y)],
        [np.sin(al_y),  np.cos(al_y)],
    ])
    X = np.dot(np.dot(x_rot, np.diag([l_x, w_x])**2), x_rot.T)
    Y = np.dot(np.dot(y_rot, np.diag([l_y, w_y]) ** 2), y_rot.T)
    X_sqrt = sqrtm(X)
    Y_sqrt = sqrtm(Y)
    return np.sum((m_x-m_y)**2) + np.sum((X_sqrt - Y_sqrt)**2)


def get_ellipse_params(ell):
    ellipse_axis, v = np.linalg.eig(ell)
    ellipse_axis = np.sqrt(ellipse_axis)
    l = ellipse_axis[0]
    w = ellipse_axis[1]
    al = np.arctan2(v[1, 0], v[0, 0])

    return l, w, al


def plot_ellipses(m_a, l_a, w_a, al_a, m_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, m_res, l_res, w_res,
                  al_res, title, name, est_color='red'):
    fig, ax = plt.subplots(1, 1)

    el_gt = Ellipse((m_gt[0], m_gt[1]), 2 * l_gt, 2 * w_gt, np.rad2deg(al_gt), fill=True, linewidth=2.0)
    el_gt.set_alpha(0.7)
    el_gt.set_fc('grey')
    el_gt.set_ec('grey')
    ax.add_artist(el_gt)

    ela_final = Ellipse((m_a[0], m_a[1]), 2 * l_a, 2 * w_a, np.rad2deg(al_a), fill=False, linewidth=2.0)
    ela_final.set_alpha(0.7)
    ela_final.set_ec('mediumpurple')
    ax.add_artist(ela_final)

    elb_final = Ellipse((m_b[0], m_b[1]), 2 * l_b, 2 * w_b, np.rad2deg(al_b), fill=False, linewidth=2.0)
    elb_final.set_alpha(0.7)
    elb_final.set_ec('darkturquoise')
    ax.add_artist(elb_final)

    el_res = Ellipse((m_res[0], m_res[1]), 2 * l_res, 2 * w_res, np.rad2deg(al_res), fill=False, linewidth=2.0)
    el_res.set_alpha(0.9)
    el_res.set_ec(est_color)
    ax.add_artist(el_res)

    plt.axis([-6, 6, -5, 7])
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.xlabel('x in m')
    plt.ylabel('y in m')
    plt.savefig(name)
    plt.show()
