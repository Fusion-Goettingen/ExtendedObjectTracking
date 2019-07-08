import numpy as np
from numpy.random import normal
from numpy.random import multivariate_normal as mvn
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

from FusionMethods.approximations import particle_approx, ord_fusion, best_or, shape_mean, shape_mean_k, lin_approx,\
    single_particle_approx_gaussian, error_and_plotting, get_jacobian, best_fusion
from FusionMethods.helpers import get_ellipse_params, plot_ellipses, gauss_wasserstein


def test_all(runs, m_a, l_a, w_a, al_a, rot_a, cov_a, cov_b, n_particles):
    error_particle = 0
    error_particle_direct = 0
    error_particle_direct_sym = 0
    error_particle_grid = 0
    error_smean = 0
    error_smean_k = 0
    error_reg = 0
    error_bestparams = 0
    error_lin = 0

    r = 0

    # test different methods ===========================================================================================
    while r < runs:
        if r % 10 == 0:
            print(r)

        # create gt and measurement from it
        m_gt = mvn(m_a, cov_a[:2, :2])
        m_b = mvn(m_gt, cov_b[:2, :2])

        l_gt = np.maximum(normal(l_a, cov_a[3, 3]), 0.1)
        w_gt = np.maximum(normal(w_a, cov_a[4, 4]), 0.1)
        al_gt = normal(al_a, cov_a[2, 2])  # - 0.1 * np.pi
        al_gt %= (2 * np.pi)
        rot_gt = np.array([
            [np.cos(al_gt), -np.sin(al_gt)],
            [np.sin(al_gt), np.cos(al_gt)],
        ])

        l_b = np.maximum(normal(w_gt, cov_b[3, 3]), 0.1)
        w_b = np.maximum(normal(l_gt, cov_b[4, 4]), 0.1)
        al_b = normal(al_gt, cov_b[2, 2]) + 0.5 * np.pi
        al_b %= (2 * np.pi)
        rot_b = np.array([
            [np.cos(al_b), -np.sin(al_b)],
            [np.sin(al_b), np.cos(al_b)],
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
        [e_p, e_d, e_s, e_g] = particle_approx(n_particles, m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b,
                                               m_gt, l_gt, w_gt, al_gt, ['gaussian', 'direct', 'direct_sym'],
                                               (r + 1) == runs)
        error_particle += e_p
        error_particle_direct += e_d
        error_particle_direct_sym += e_s
        error_particle_grid += e_g

        # apply ordinary fusion on the originals
        error_reg += ord_fusion(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                                (r + 1) == runs)

        # apply ordinary fusion on the originals with the best rotation
        error_bestparams += best_or(m_a, cov_a, l_a, w_a, al_a, m_b, cov_b, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                                    (r + 1) == runs)

        # shape mean
        error_smean += shape_mean(m_a, A, l_a, w_a, al_a, m_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                                  (r + 1) == runs)

        # linearization
        error_lin += lin_approx(m_a, cov_a, A, l_a, w_a, al_a, m_b, cov_b, B, l_b, w_b, al_b, m_gt, l_gt, w_gt, al_gt,
                                (r + 1) == runs)

        r += 1

    # visualize and print error ========================================================================================
    bars = np.array([1, 3, 5, 7, 9, 11, 13])
    ticks = np.array(['Regular', 'Mean of\n shape matrix', 'MMGW-Lin', 'Heuristic', 'MMGW-MC', 'MC GW dir',
                      'MC GW sym'])
    plt.bar(1, np.sqrt(error_reg / runs), width=0.5, color='red', align='center')
    plt.bar(3, np.sqrt(error_smean / runs), width=0.5, color='m', align='center')
    plt.bar(5, np.sqrt(error_lin / runs), width=0.5, color='deepskyblue', align='center')
    plt.bar(7, np.sqrt(error_bestparams / runs), width=0.5, color='darkcyan', align='center')
    plt.bar(9, np.sqrt(error_particle / runs), width=0.5, color='darkgreen', align='center')
    plt.bar(11, np.sqrt(error_particle_direct / runs), width=0.5, color='green', align='center')
    plt.bar(13, np.sqrt(error_particle_direct_sym / runs), width=0.5, color='lightgreen', align='center')
    plt.xticks(bars, ticks)
    plt.title('GW RMSE')
    plt.ylim(0.0, 1.5)
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
    print('MC fuse direct:')
    print(np.sqrt(error_particle_direct / runs))
    print('MC fuse direct sym:')
    print(np.sqrt(error_particle_direct_sym / runs))


def test_convergence(steps, m_prior, l_prior, w_prior, al_prior, cov_a, cov_b, n_particles):
    save_path = '/home/convergence/'
    error = np.zeros(steps)
    error_reg = np.zeros(steps)

    # create gt with sensor A as prior
    m_gt = m_prior  # mvn(m_a, cov_a[:2, :2])

    l_gt = np.maximum(normal(l_prior, cov_a[3, 3]), 0.1)
    w_gt = np.maximum(normal(w_prior, cov_a[4, 4]), 0.1)  # np.maximum(normal(w_gt, cov_a[4, 4]), 0.1)
    al_gt = normal(al_prior, cov_a[2, 2])  # - 0.1 * np.pi
    al_gt %= (2 * np.pi)

    # get prior in square root space
    m_prior, el_sr_prior, cov_sr_prior = single_particle_approx_gaussian(m_prior, l_prior, w_prior, al_prior, cov_a,
                                                                         n_particles, False)

    # get prior for regular state
    m_prior_reg = m_prior.copy()
    prior_reg = np.array([al_prior, l_prior, w_prior])
    cov_prior_reg = cov_a.copy()[2:, 2:]

    # test different methods ===========================================================================================
    for i in range(steps):
        if i % 10 == 0:
            print(i)

        # create measurement from gt (using alternating sensors)
        if (i % 2) == 0:
            m_m = m_gt  # mvn(m_gt, cov_b[:2, :2])
            l_m = np.maximum(normal(w_gt, cov_b[3, 3]), 0.1)
            w_m = np.maximum(normal(l_gt, cov_b[4, 4]), 0.1)
            al_m = normal(al_gt, cov_b[2, 2]) + 0.5 * np.pi
            al_m %= (2 * np.pi)
            cov_m = cov_b.copy()
        else:
            m_m = m_gt  # mvn(m_gt, cov_a[:2, :2])
            l_m = np.maximum(normal(l_gt, cov_a[3, 3]), 0.1)
            w_m = np.maximum(normal(w_gt, cov_a[4, 4]), 0.1)
            al_m = normal(al_gt, cov_a[2, 2])
            al_m %= (2 * np.pi)
            cov_m = cov_a.copy()

        # MC approximation =============================================================================================

        # get measurement in square root space
        m_m, el_sr_m, cov_sr_m = single_particle_approx_gaussian(m_m, l_m, w_m, al_m, cov_m, n_particles, False)

        # get posterior in square root space
        # fuse A and B via mean and variance
        S = cov_sr_prior + cov_sr_m
        K = np.dot(cov_sr_prior, np.linalg.inv(S))
        el_sr_post = el_sr_prior + np.dot(K, el_sr_m - el_sr_prior)
        cov_sr_post = cov_sr_prior - np.dot(np.dot(K, S), K.T)

        # var_Res = var_A - np.dot(np.dot(K, S), K.T)
        m_post = 0.5 * (m_prior + m_m)  # mean_Res[:2]
        el_sr_res = np.array([
            [el_sr_post[0], el_sr_post[1]],
            [el_sr_post[1], el_sr_post[2]],
        ])
        el_res = np.dot(el_sr_res, el_sr_res)

        error[i] = error_and_plotting(m_post, el_res, m_prior, l_prior, w_prior, al_prior, m_m, l_m, w_m, al_m, m_gt,
                                      l_gt, w_gt, al_gt, True, 'MC Approximated Fusion',
                                      save_path + 'exampleMCApprox%i.svg' % i)

        m_prior = m_post
        el_sr_prior = el_sr_post
        cov_sr_prior = cov_sr_post

        l_prior, w_prior, al_prior = get_ellipse_params(el_res)

        # regular fusion ===============================================================================================

        # get posterior in square root space
        # fuse A and B via mean and variance
        S = cov_prior_reg + cov_m[2:, 2:]
        K = np.dot(cov_prior_reg, np.linalg.inv(S))
        post_reg = prior_reg + np.dot(K, np.array([al_m, l_m, w_m]) - prior_reg)
        cov_post_reg = cov_prior_reg - np.dot(np.dot(K, S), K.T)

        m_post_reg = 0.5 * (m_prior_reg + m_m)

        plot_ellipses(m_prior, prior_reg[1], prior_reg[2], prior_reg[0], m_m, l_m, w_m, al_m, m_gt, l_gt, w_gt, al_gt,
                      m_post_reg, post_reg[1], post_reg[2], post_reg[0], 'Fusion of Original State',
                      save_path + 'exampleRegFus%i.svg' % i)

        error_reg[i] = gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, m_post_reg, post_reg[1], post_reg[2], post_reg[0])

        m_prior_reg = m_post_reg
        prior_reg = post_reg
        cov_prior_reg = cov_post_reg

    plt.plot(np.arange(1, steps+1), error)
    plt.savefig(save_path + 'MCApproxError.svg')
    plt.show()
    plt.plot(np.arange(1, steps + 1), error_reg)
    plt.savefig(save_path + 'regFusError.svg')
    plt.show()
    plt.plot(np.arange(1, steps + 1), error, color='green', label='MC Fusion')
    plt.plot(np.arange(1, steps + 1), error_reg, color='red', label='Regular Fusion')
    plt.legend()
    plt.savefig(save_path + 'errorComp.svg')
    plt.show()


def test_convergence_pos(steps, runs, m_prior, l_prior, w_prior, al_prior, cov_a, cov_b, n_particles):
    save_path = '/home/kthormann/Desktop/research-fusion/research-fusion/Python/EllipseFusion/convergence/'
    error = np.zeros(steps)
    error_reg = np.zeros(steps)
    error_shape = np.zeros(steps)
    error_params = np.zeros(steps)
    error_lin = np.zeros(steps)

    for r in range(runs):
        print('Run %i of %i' % (r, runs))
        # create gt with sensor A as prior
        m_gt = mvn(m_prior, cov_a[:2, :2])

        l_gt = np.maximum(normal(l_prior, cov_a[3, 3]), 0.1)
        w_gt = np.maximum(normal(w_prior, cov_a[4, 4]), 0.1)  # np.maximum(normal(w_gt, cov_a[4, 4]), 0.1)
        al_gt = normal(al_prior, cov_a[2, 2])  # - 0.1 * np.pi
        al_gt %= (2 * np.pi)

        # get prior in square root space
        m_prior, el_sr_prior, cov_sr_prior = single_particle_approx_gaussian(m_prior, l_prior, w_prior, al_prior, cov_a,
                                                                             n_particles, True)

        # get prior for regular state
        prior_reg = np.array([m_prior[0], m_prior[1], al_prior, l_prior, w_prior])
        cov_prior_reg = cov_a.copy()

        # get prior for best params
        prior_params = np.copy(prior_reg)
        cov_prior_params = np.copy(cov_prior_reg)

        # get prior for shape mean
        prior_shape_m = m_prior
        prior_shape_l = l_prior
        prior_shape_w = w_prior
        prior_shape_al = al_prior
        rot_prior = np.array([
            [np.cos(al_prior), -np.sin(al_prior)],
            [np.sin(al_prior),  np.cos(al_prior)],
        ])
        prior_shape = np.dot(np.dot(rot_prior, np.diag([l_prior, w_prior])**2), rot_prior.T)

        # get prior for linearization
        prior_shape_sqrt = sqrtm(prior_shape)
        prior_lin = np.array([m_prior[0], m_prior[1], prior_shape_sqrt[0, 0], prior_shape_sqrt[1, 0],
                              prior_shape_sqrt[1, 1]])
        prior_lin_l = l_prior
        prior_lin_w = w_prior
        prior_lin_al = al_prior
        jac = get_jacobian(l_prior, w_prior, al_prior)
        prior_lin_cov = np.dot(np.dot(jac, cov_prior_reg), jac.T)

        # test different methods =======================================================================================
        for i in range(steps):
            if i % 10 == 0:
                print(i)

            # create measurement from gt (using alternating sensors)
            if (i % 2) == 0:
                m_m = mvn(m_gt, cov_b[:2, :2])
                l_m = np.maximum(normal(w_gt, cov_b[3, 3]), 0.1)
                w_m = np.maximum(normal(l_gt, cov_b[4, 4]), 0.1)
                al_m = normal(al_gt, cov_b[2, 2]) + 0.5 * np.pi
                al_m %= (2 * np.pi)
                cov_m = cov_b.copy()
            else:
                m_m = mvn(m_gt, cov_a[:2, :2])
                l_m = np.maximum(normal(l_gt, cov_a[3, 3]), 0.1)
                w_m = np.maximum(normal(w_gt, cov_a[4, 4]), 0.1)
                al_m = normal(al_gt, cov_a[2, 2])
                al_m %= (2 * np.pi)
                cov_m = cov_a.copy()

            # MC approximation =========================================================================================

            # get measurement in square root space
            m_m, el_sr_m, cov_sr_m = single_particle_approx_gaussian(m_m, l_m, w_m, al_m, cov_m, n_particles, True)

            # get posterior in square root space
            # fuse A and B via mean and variance
            S = cov_sr_prior + cov_sr_m
            K = np.dot(cov_sr_prior, np.linalg.inv(S))
            el_sr_post = el_sr_prior + np.dot(K, el_sr_m - el_sr_prior)
            cov_sr_post = cov_sr_prior - np.dot(np.dot(K, S), K.T)

            # var_Res = var_A - np.dot(np.dot(K, S), K.T)
            el_sr_res = np.array([
                [el_sr_post[2], el_sr_post[3]],
                [el_sr_post[3], el_sr_post[4]],
            ])
            el_res = np.dot(el_sr_res, el_sr_res)

            error[i] += error_and_plotting(el_sr_post[:2], el_res, el_sr_prior[:2], l_prior, w_prior, al_prior, m_m, l_m,
                                          w_m, al_m, m_gt, l_gt, w_gt, al_gt, (r + 1) == runs, 'MC Approximated Fusion',
                                          save_path + 'exampleMCApprox%i.svg' % i, est_color='green')

            el_sr_prior = el_sr_post
            cov_sr_prior = cov_sr_post

            l_prior, w_prior, al_prior = get_ellipse_params(el_res)

            # regular fusion ===========================================================================================

            # get posterior in square root space
            # fuse A and B via mean and variance
            S = cov_prior_reg + cov_m
            K = np.dot(cov_prior_reg, np.linalg.inv(S))
            post_reg = prior_reg + np.dot(K, np.array([m_m[0], m_m[1], al_m, l_m, w_m]) - prior_reg)
            cov_post_reg = cov_prior_reg - np.dot(np.dot(K, S), K.T)

            if (r + 1) == runs:
                plot_ellipses(prior_reg[:2], prior_reg[3], prior_reg[4], prior_reg[2], m_m, l_m, w_m, al_m, m_gt, l_gt,
                              w_gt, al_gt, post_reg[:2], post_reg[3], post_reg[4], post_reg[2],
                              'Fusion of Original State', save_path + 'exampleRegFus%i.svg' % i)

            error_reg[i] += gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, post_reg[:2], post_reg[3], post_reg[4],
                                              post_reg[2])

            prior_reg = post_reg
            cov_prior_reg = cov_post_reg

            # best params fusion =======================================================================================
            [post_params, cov_post_params, k] = best_fusion(prior_params[:2], cov_prior_params, prior_params[3],
                                                            prior_params[4], prior_params[2], m_m, cov_m, l_m, w_m,
                                                            al_m)

            if (r + 1) == runs:
                plot_ellipses(prior_params[:2], prior_params[3], prior_params[4], prior_params[2], m_m, l_m, w_m, al_m,
                              m_gt, l_gt, w_gt, al_gt, post_params[:2], post_params[3], post_params[4], post_params[2],
                              'Best Params', save_path + 'exampleBestParams%i.svg' % i)

            error_params[i] += gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, post_params[:2], post_params[3],
                                                 post_params[4], post_params[2])

            prior_params = post_params
            cov_prior_params = cov_post_params

            # shape mean ===============================================================================================
            post_shape_m = 0.5 * (prior_shape_m + m_m)
            rot_m = np.array([
                [np.cos(al_m), -np.sin(al_m)],
                [np.sin(al_m),  np.cos(al_m)],
            ])
            m_shape = np.dot(np.dot(rot_m, np.diag([l_m, w_m]) ** 2), rot_m.T)
            post_shape = 0.5 * (prior_shape + m_shape)

            post_shape_l, post_shape_w, post_shape_al = get_ellipse_params(post_shape)

            if (r + 1) == runs:
                plot_ellipses(prior_shape_m, prior_shape_l, prior_shape_w, prior_shape_al, m_m, l_m, w_m, al_m, m_gt,
                              l_gt, w_gt, al_gt, post_shape_m, post_shape_l, post_shape_w, post_shape_al, 'Shape Mean', 'exampleMean.svg')

            error_shape[i] += gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, post_shape_m, post_shape_l, post_shape_w,
                                                post_shape_al)

            prior_shape_m = post_shape_m
            prior_shape = post_shape
            prior_shape_l = post_shape_l
            prior_shape_w = post_shape_w
            prior_shape_al = post_shape_al

            # linearization ============================================================================================
            m_shape_sqrt = sqrtm(m_shape)
            lin_m = np.array([m_m[0], m_m[1], m_shape_sqrt[0, 0], m_shape_sqrt[1, 0], m_shape_sqrt[1, 1]])
            jac_m = get_jacobian(l_m, w_m, al_m)
            cov_m_lin = np.dot(np.dot(jac_m, cov_m), jac_m.T)
            S_lin = prior_lin_cov + cov_m_lin
            K_lin = np.dot(prior_lin_cov, np.linalg.inv(S_lin))
            post_lin = prior_lin + np.dot(K_lin, lin_m - prior_lin)
            post_lin_cov = prior_lin_cov - np.dot(np.dot(K_lin, S_lin), K_lin.T)

            post_lin_shape = np.array([
                [post_lin[2], post_lin[3]],
                [post_lin[3], post_lin[4]],
            ])
            post_lin_shape = np.dot(post_lin_shape, post_lin_shape)
            ellipse_axis_post_lin, v_post_lin = np.linalg.eig(post_lin_shape)
            ellipse_axis_post_lin = np.sqrt(ellipse_axis_post_lin)
            post_lin_l, post_lin_w, post_lin_al = get_ellipse_params(post_lin_shape)

            if (r + 1) == runs:
                plot_ellipses(prior_lin[:2], prior_lin_l, prior_lin_w, prior_lin_al, m_m, l_m, w_m, al_m, m_gt, l_gt,
                              w_gt, al_gt, post_lin[:2], post_lin_l, post_lin_w, post_lin_al, 'Shape Mean',
                              'exampleMean.svg')

            error_lin[i] += gauss_wasserstein(m_gt, l_gt, w_gt, al_gt, post_lin[:2], post_lin_l, post_lin_w, post_lin_al)

            prior_lin = post_lin
            prior_lin_cov = post_lin_cov
            prior_lin_l = post_lin_l
            prior_line_w = post_lin_w
            prior_lin_al = post_lin_al

    error = np.sqrt(error / runs)
    error_reg = np.sqrt(error_reg / runs)
    error_shape = np.sqrt(error_shape / runs)
    error_params = np.sqrt(error_params / runs)
    error_lin = np.sqrt(error_lin / runs)

    print(error)
    print(error_params)

    # error first and last
    bars = np.array([1, 3, 5, 7, 9])
    ticks = np.array(['Regular', 'Mean of\n shape matrix', 'MMGW-Lin', 'Heuristic', 'MMGW-MC'])
    plt.bar(0.825, error_reg[0], width=0.25, color='red', align='center')
    plt.bar(1.175, error_reg[-1], width=0.25, color='red', align='center')
    plt.bar(2.825, error_shape[0], width=0.25, color='m', align='center')
    plt.bar(3.175, error_shape[-1], width=0.25, color='m', align='center')
    plt.bar(4.825, error_lin[0], width=0.25, color='deepskyblue', align='center')
    plt.bar(5.175, error_lin[-1], width=0.25, color='deepskyblue', align='center')
    plt.bar(6.825, error_params[0], width=0.25, color='darkcyan', align='center')
    plt.bar(7.175, error_params[-1], width=0.25, color='darkcyan', align='center')
    plt.bar(8.825, error[0], width=0.25, color='green', align='center')
    plt.bar(9.175, error[-1], width=0.25, color='green', align='center')
    plt.xticks(bars, ticks)
    plt.title('GW RMSE')
    plt.ylim(0.0, 1.5)
    plt.savefig('gwRmse.svg')
    plt.show()

    plt.plot(np.arange(1, steps+1), error)
    plt.savefig(save_path + 'MCApproxError.svg')
    plt.show()
    plt.plot(np.arange(1, steps + 1), error_reg)
    plt.savefig(save_path + 'regFusError.svg')
    plt.show()
    plt.plot(np.arange(1, steps + 1), error_reg, color='red', label='Regular State')
    plt.plot(np.arange(1, steps + 1), error_shape, color='m', label='Mean of shape matrix')
    plt.plot(np.arange(1, steps + 1), error_lin, color='deepskyblue', label='MMGW-Lin')
    plt.plot(np.arange(1, steps + 1), error_params, color='darkcyan', label='Heuristic')
    plt.plot(np.arange(1, steps + 1), error, color='green', label='MMGW-MC')
    plt.legend()
    plt.xticks([5, 10, 15, 20])
    plt.savefig(save_path + 'errorComp.svg')
    plt.show()
