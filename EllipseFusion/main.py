"""
Evaluation for
K. Thormann and M. Baum, "Optimal Fusion of Elliptic Extended Target Estimates based on the Wasserstein Distance",
 ArXix e-prints, April 2019 (online). Available: https://arxiv.org/abs/1904.00708

Author: Kolja Thormann
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


from FusionMethods.helpers import *
from FusionMethods.approximations import *


# setup ================================================================================================================
# Ground truth
m_gt = np.array([0, 1])

l_gt = 4
w_gt = 2
al_gt = np.pi * 0.5
rot_gt = np.array([
    [np.cos(al_gt), -np.sin(al_gt)],
    [np.sin(al_gt), np.cos(al_gt)]
])

Gt = np.dot(np.dot(rot_gt, np.diag([l_gt, w_gt]) ** 2), rot_gt.T)

# noisy estimates of sensors A and B
cov_a = np.array([
    [0.5, 0.0, 0, 0, 0],
    [0.0, 0.5, 0, 0, 0],
    [0, 0, 0.2, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0.2],
])
cov_b = np.array([
    [1.5, 0.0, 0, 0, 0],
    [0.0, 1.5, 0, 0, 0],
    [0, 0, 0.2, 0, 0],
    [0, 0, 0, 1, 0],  # will be w
    [0, 0, 0, 0, 0.2],  # will be l
])

error_particle = 0
error_smean = 0
error_reg = 0
error_bestparams = 0
error_lin = 0

runs = 1
r = 0

# test different methods ===============================================================================================
while r < runs:
    if r % 10 == 0:
        print(r)

    # create two estimates
    m_a = mvn(m_gt, cov_a[:2, :2])
    m_b = mvn(m_gt, cov_b[:2, :2])

    l_a = np.maximum(normal(l_gt, cov_a[3, 3]), 0.1)
    w_a = np.maximum(normal(w_gt, cov_a[4, 4]), 0.1)
    al_a = normal(al_gt, cov_a[2, 2])
    al_a %= (2 * np.pi)
    rot_a = np.array([
        [np.cos(al_a), -np.sin(al_a)],
        [np.sin(al_a), np.cos(al_a)]
    ])

    l_b = np.maximum(normal(w_gt, cov_b[3, 3]), 0.1)
    w_b = np.maximum(normal(l_gt, cov_b[4, 4]), 0.1)
    al_b = normal(al_gt, cov_b[2, 2]) + 0.5 * np.pi
    al_b %= (2 * np.pi)
    rot_b = np.array([
        [np.cos(al_b), -np.sin(al_b)],
        [np.sin(al_b), np.cos(al_b)]
    ])

    A = np.dot(np.dot(rot_a, np.diag([l_a, w_a]) ** 2), rot_a.T)
    B = np.dot(np.dot(rot_b, np.diag([l_b, w_b]) ** 2), rot_b.T)

    if (not all(np.linalg.eigvals(A) > 0)):
        print('A is not positiv definite')
        # print(A)
        continue
    elif A.dtype != np.float64:
        print('A is complex')
        # print(A)
        continue
    if (not all(np.linalg.eigvals(B) > 0)):
        print('B is not positiv definite')
        # print(B)
        continue
    elif B.dtype != np.float64:
        print('B is complex')
        # print(B)
        continue

    # use particles to calculate mean and variance of A nd B ellipse
    n_particles = 10000
    error_particle += particle_approx(n_particles, m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt,
                                      w_gt, al_gt, (r+1) == runs)

    # apply ordinary fusion on the originals
    error_reg += ord_fusion(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                            (r+1) == runs)

    # apply ordinary fusion on the originals with the best rotation
    error_bestparams += best_or(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                                (r+1) == runs)

    # shape mean
    error_smean += shape_mean(m_a, A, l_a, w_a, al_a, m_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt, (r+1) == runs)

    # linearization
    error_lin += lin_approx(m_a, cov_a, A, l_a, w_a, al_a, m_b, cov_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                            (r+1) == runs)

    r += 1

# visualize and print error ============================================================================================
bars = np.array([1, 3, 5, 7, 9])
ticks = np.array(['Regular', 'Shape Mean', 'Approx GW', 'Best Params', 'MC GW'])
plt.bar(1, np.sqrt(error_reg / runs), width=0.5, color='red', align='center')
plt.bar(3, np.sqrt(error_smean / runs), width=0.5, color='m', align='center')
plt.bar(5, np.sqrt(error_lin / runs), width=0.5, color='deepskyblue', align='center')
plt.bar(7, np.sqrt(error_bestparams / runs), width=0.5, color='darkcyan', align='center')
plt.bar(9, np.sqrt(error_particle / runs), width=0.5, color='green', align='center')
plt.xticks(bars, ticks)
plt.title('GW RMSE')
plt.ylim(0.7, 1.4)
plt.savefig('gwRmse.svg')
plt.show()

print('Fuse of originals:')
print(np.sqrt(error_reg / runs))
print('Mean of originals:')
print(np.sqrt(error_smean / runs))
print('Lin Fuse:')
print(np.sqrt(error_lin / runs))
print('Fuse of originals with best orientation:')
print(np.sqrt(error_bestparams / runs))
print('MC fuse:')
print(np.sqrt(error_particle / runs))
