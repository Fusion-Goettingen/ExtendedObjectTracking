"""
Author: Kolja Thormann

Evaluation for
Thormann, Kolja; Baum, Marcus (2019): Bayesian Fusion of Orientation, Width and Length Estimates of Elliptic
Extended Targets based on the Gaussian Wasserstein Distance. TechRxiv. Preprint.
"""

import numpy as np

from FusionMethods.tests import test_convergence_pos, test_mean


# setup ================================================================================================================
# Gaussian prior
prior = np.array([0, 0, 0.0 * np.pi, 8, 3])  # [m1, m2, alpha, length, width]
cov_prior = np.array([
    [0.5, 0.0, 0,         0,     0],
    [0.0, 0.5, 0,         0,     0],
    [0,   0,   0.5*np.pi, 0,     0],
    [0,   0,   0,         0.5,   0],
    [0,   0,   0,         0,   0.5],
])

# sensor A and B noise
cov_a = np.array([
    [0.5, 0.0, 0,         0,     0],
    [0.0, 0.5, 0,         0,     0],
    [0,   0,   0.01*np.pi, 0,     0],
    [0,   0,   0,         0.5,   0],
    [0,   0,   0,         0,   0.1],
])
cov_b = np.array([
    [0.5, 0.0, 0,         0,     0],
    [0.0, 0.5, 0,         0,     0],
    [0,   0,   0.01*np.pi, 0,     0],
    [0,   0,   0,         0.5,   0],  # will be w
    [0,   0,   0,         0,   0.1],  # will be l
])

runs = 100  # number of MC runs
steps = 20  # number of measurements (alternating sensors A and B)
n_particles = 1000  # number of particles for MMSR-MC
n_particles_pf = 100000  # number of particles for MMSR-PF

save_path = './'

# tests ================================================================================================================
test_convergence_pos(steps, runs, prior, cov_prior, cov_a, cov_b, n_particles, n_particles_pf, save_path)
# test_mean(prior, cov_prior, n_particles, save_path)
