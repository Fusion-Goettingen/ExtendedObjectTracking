"""
Author: Kolja Thormann

Contains functions for calculating and plotting the error and ellipses
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.linalg import sqrtm
from matplotlib.patches import Ellipse

from FusionMethods.ellipse_fusion_support import rot_matrix, to_matrix


def error_and_plotting(m_res, l_res, w_res, al_res, m_prior, l_prior, w_prior, al_prior, m_meas, l_meas, w_meas,
                       al_meas, m_gt, l_gt, w_gt, al_gt, plotting, name, filename, est_color='red'):
    """
    If demanded, plot ellipse prior, estimate, measurement, and ground truth and provide the error as GW and SR distance
    :param m_res:       Center of estimated ellipse
    :param l_res:       Semi-axis length of estimated ellipse
    :param w_res:       Semi-axis width of estimated ellipse
    :param al_res:      Orientation of estimated ellipse
    :param m_prior:     Center of prior ellipse
    :param l_prior:     Semi-axis length of prior ellipse
    :param w_prior:     Semi-axis width of prior ellipse
    :param al_prior:    Orientation of prior ellipse
    :param m_meas:      Center of measured ellipse
    :param l_meas:      Semi-axis length of measured ellipse
    :param w_meas:      Semi-axis width of measured ellipse
    :param al_meas:     Orientation of measured ellipse
    :param m_gt:        Center of ground truth ellipse
    :param l_gt:        Semi-axis length of ground truth ellipse
    :param w_gt:        Semi-axis width of ground truth ellipse
    :param al_gt:       Orientation of ground truth ellipse
    :param plotting:    Boolean to determine whether to plot the ellipses
    :param name:        Name of the approach
    :param filename:    Name of the file the plot is to be saved in
    :param est_color:   Color for plotting the estimated ellipse
    :return:            Error of the estimate in Gaussian Wasserstein and Square Root distance
    """
    if plotting:
        plot_ellipses(m_res, l_res, w_res, al_res, m_prior, l_prior, w_prior, al_prior, m_meas, l_meas, w_meas, al_meas,
                      m_gt, l_gt, w_gt, al_gt, name, filename, est_color)

    return [gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, m_res, l_res, w_res, al_res),
            square_root_distance(m_gt, l_gt, w_gt, al_gt, m_res, l_res, w_res, al_res)]


def gauss_wasserstein(m_x, l_x, w_x, al_x, m_y, l_y, w_y, al_y):
    """
    Calculate the Gaussian Wasserstein distance of two ellipses
    :param m_x:     Center of first ellipse
    :param l_x:     Semi-axis length of first ellipse
    :param w_x:     Semi-axis width of first ellipse
    :param al_x:    Orientation of first ellipse
    :param m_y:     Center of second ellipse
    :param l_y:     Semi-axis length of second ellipse
    :param w_y:     Semi-axis width of second ellipse
    :param al_y:    Orientation of second ellipse
    :return:        The Gaussian Wasserstein distance of the two ellipses
    """
    gt_xc = m_x
    gt_sigma = to_matrix(al_x, l_x, w_x, False)
    gt_sigma += gt_sigma.T
    gt_sigma /= 2

    track_xc = m_y
    track_sigma = to_matrix(al_y, l_y, w_y, False)
    track_sigma += track_sigma.T
    track_sigma /= 2

    error = norm(gt_xc - track_xc) ** 2 + np.trace(gt_sigma + track_sigma
                                                   - 2 * sqrtm(np.einsum('ab, bc, cd -> ad', sqrtm(gt_sigma),
                                                                         track_sigma, sqrtm(gt_sigma))))

    return error


def square_root_distance(m_x, l_x, w_x, al_x, m_y, l_y, w_y, al_y):
    """
    Calculate the Square Root distance of two ellipses
    :param m_x:     Center of first ellipse
    :param l_x:     Semi-axis length of first ellipse
    :param w_x:     Semi-axis width of first ellipse
    :param al_x:    Orientation of first ellipse
    :param m_y:     Center of second ellipse
    :param l_y:     Semi-axis length of second ellipse
    :param w_y:     Semi-axis width of second ellipse
    :param al_y:    Orientation of second ellipse
    :return:        The Square Root distance of the two ellipses
    """
    x_rot = rot_matrix(al_x)
    y_rot = rot_matrix(al_y)
    X_sqrt = np.dot(np.dot(x_rot, np.diag([l_x, w_x])), x_rot.T)
    Y_sqrt = np.dot(np.dot(y_rot, np.diag([l_y, w_y])), y_rot.T)

    return np.sum((m_x-m_y)**2) + np.sum((X_sqrt - Y_sqrt)**2)


def plot_ellipses(m_res, l_res, w_res, al_res, m_prior, l_prior, w_prior, al_prior, m_meas, l_meas, w_meas, al_meas,
                  m_gt, l_gt, w_gt, al_gt, name, filename, est_color='red'):
    """
    Plot the estimated, prior, measured, and ground truth ellipses and save the plot
    :param m_res:       Center of estimated ellipse
    :param l_res:       Semi-axis length of estimated ellipse
    :param w_res:       Semi-axis width of estimated ellipse
    :param al_res:      Orientation of estimated ellipse
    :param m_prior:     Center of prior ellipse
    :param l_prior:     Semi-axis length of prior ellipse
    :param w_prior:     Semi-axis width of prior ellipse
    :param al_prior:    Orientation of prior ellipse
    :param m_meas:      Center of measured ellipse
    :param l_meas:      Semi-axis length of measured ellipse
    :param w_meas:      Semi-axis width of measured ellipse
    :param al_meas:     Orientation of measured ellipse
    :param m_gt:        Center of ground truth ellipse
    :param l_gt:        Semi-axis length of ground truth ellipse
    :param w_gt:        Semi-axis width of ground truth ellipse
    :param al_gt:       Orientation of ground truth ellipse
    :param name:        Name of the approach
    :param filename:    Name of the file the plot is to be saved in
    :param est_color:   Color for plotting the estimated ellipse
    """
    fig, ax = plt.subplots(1, 1)

    el_gt = Ellipse((m_gt[0], m_gt[1]), 2 * l_gt, 2 * w_gt, np.rad2deg(al_gt), fill=True, linewidth=2.0)
    el_gt.set_alpha(0.7)
    el_gt.set_fc('grey')
    el_gt.set_ec('grey')
    ax.add_artist(el_gt)

    ela_final = Ellipse((m_prior[0], m_prior[1]), 2 * l_prior, 2 * w_prior, np.rad2deg(al_prior), fill=False,
                        linewidth=2.0)
    ela_final.set_alpha(0.7)
    ela_final.set_ec('mediumpurple')
    ax.add_artist(ela_final)

    elb_final = Ellipse((m_meas[0], m_meas[1]), 2 * l_meas, 2 * w_meas, np.rad2deg(al_meas), fill=False, linewidth=2.0)
    elb_final.set_alpha(0.7)
    elb_final.set_ec('darkturquoise')
    ax.add_artist(elb_final)

    el_res = Ellipse((m_res[0], m_res[1]), 2 * l_res, 2 * w_res, np.rad2deg(al_res), fill=False, linewidth=2.0)
    el_res.set_alpha(0.9)
    el_res.set_ec(est_color)
    ax.add_artist(el_res)

    plt.axis([-10, 10, -10, 10])
    ax.set_aspect('equal')
    ax.set_title(name)
    plt.xlabel('x in m')
    plt.ylabel('y in m')
    plt.savefig(filename)
    plt.show()


def plot_error_bars(states, steps):
    """
    For each approach stored in states, plot the error of the first and last measurement step as a bar chart; error in
    GW and SR distance
    :param states:  Contains names and errors of approaches; expects first steps entries of error to be in GW distance
                    and last steps entries in SR distance
    :param steps:   Number of measurement steps
    """
    # setup
    num_states = len(states)
    bars = np.arange(1, 2 * num_states, 2)
    ticks = states[:]['name']

    # plot Gw distance
    for i in range(num_states):
        plt.bar(bars[i] - 0.175, states[i]['error'][0], width=0.25, color=states[i]['color'], align='center')
        plt.bar(bars[i] + 0.175, states[i]['error'][steps - 1], width=0.25, color=states[i]['color'], align='center')
    plt.xticks(bars, ticks)
    plt.title('GW RMSE')
    plt.savefig('gwRmse.svg')
    plt.show()

    # plot SR distance
    for i in range(num_states):
        plt.bar(bars[i] - 0.175, states[i]['error'][steps], width=0.25, color=states[i]['color'], align='center')
        plt.bar(bars[i] + 0.175, states[i]['error'][-1], width=0.25, color=states[i]['color'], align='center')
    plt.xticks(bars, ticks)
    plt.title('SR RMSE')
    plt.savefig('srRmse.svg')
    plt.show()


def plot_convergence(states, steps, save_path):
    """
    For each approach stored in states, plot the error convergence; error in GW and SR distance
    :param states:      Contains names and errors of approaches; expects first steps entries of error to be in GW
                        distance and last steps entries in SR distance
    :param steps:       Number of measurement steps
    :param save_path:   Path in which to save the plots
    """
    num_states = len(states)

    # plot GW distance
    for i in range(num_states):
        plt.plot(np.arange(1, steps + 1), states[i]['error'][:steps], color=states[i]['color'], label=states[i]['name'])
    plt.legend()
    plt.xticks(np.arange(5, steps + 1, 5))
    plt.savefig(save_path + 'errorCompGW.svg')
    plt.show()

    # plot SR distance
    for i in range(num_states):
        plt.plot(np.arange(1, steps + 1), states[i]['error'][steps:], color=states[i]['color'], label=states[i]['name'])
    plt.legend()
    plt.xticks(np.arange(5, steps + 1, 5))
    plt.savefig(save_path + 'errorCompSR.svg')
    plt.show()
