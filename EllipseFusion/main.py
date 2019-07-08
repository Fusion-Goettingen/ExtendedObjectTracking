"""
Evaluation for
K. Thormann and M. Baum, "Optimal Fusion of Elliptic Extended Target Estimates based on the Wasserstein Distance",
 ArXix e-prints, April 2019 (online). Available: https://arxiv.org/abs/1904.00708

Author: Kolja Thormann
"""

import numpy as np

from FusionMethods.tests import test_all, test_convergence, test_convergence_pos


# setup ================================================================================================================
# Prior
m_a = np.array([0, 1])

l_a = 4
w_a = 3
al_a = np.pi * 0.5
rot_a = np.array([
    [np.cos(al_a), -np.sin(al_a)],
    [np.sin(al_a), np.cos(al_a)]
])

# noisy estimates of sensors A and B
cov_a = np.array([
    [0.5, 0.0, 0, 0, 0],
    [0.0, 0.5, 0, 0, 0],
    [0, 0, 0.01*np.pi, 0, 0],
    [0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0.1],
])
cov_b = np.array([
    [0.5, 0.0, 0, 0, 0],
    [0.0, 0.5, 0, 0, 0],
    [0, 0, 0.01*np.pi, 0, 0],
    [0, 0, 0, 0.5, 0],  # will be w
    [0, 0, 0, 0, 0.1],  # will be l
])

runs = 1
steps = 20
n_particles = 1000

test_all(runs, m_a, l_a, w_a, al_a, rot_a, cov_a, cov_b, n_particles)
# test_convergence_pos(steps, runs, m_a, l_a, w_a, al_a, cov_a, cov_b, n_particles)
