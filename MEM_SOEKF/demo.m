% Matlab demo for the paper
% "Second-Order Extended Kalman Filter for Extended Object and Group Tracking", 
% Shishan Yang, Marcus Baum
% https://arxiv.org/abs/1604.00219


close all
clear
dbstop error

% generate ground truth
[gt_center, gt_rotation, gt_orient, gt_length, gt_vel, time_steps, time_interval] =get_ground_truth;
gt = [gt_center;gt_orient;gt_length;gt_vel];

motionmodel = {'NCV'};
process_matrix = [1 0 0 0 0 10 0; 0 1 0 0 0 0 10; 0 0 1 0 0 0 0; ...
    0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1];

% spread of multiplicative error
h1_var = 1/4;
h2_var = 1/4;
multi_noise_cov = diag([h1_var, h2_var]);
possion_lambda = 5;

meas_noise = 0.02*repmat(diag([100^2, 20^2]), 1, 1, time_steps);

%% -------------Setting prior---------------------------------------
initial_guess_center = [100, 100];
initial_guess_alpha = -pi/3;
initial_guess_len = [200, 90];
initial_guess_velo = 10;
initial_guess = [initial_guess_center, initial_guess_alpha, initial_guess_len, ...
    initial_guess_velo, initial_guess_velo*tan(initial_guess_alpha)]';

initial_guess_center_cov = 900*eye(2);
initial_guess_alpha_cov = 0.02*eye(1);
initial_guess_len_cov = 400*eye(2);
initial_guess_velo_cov = 16*eye(2);

mem_process_noise = blkdiag(100*eye(2), 0.05, 0.00001*eye(2), eye(2));

mem_x_pre = initial_guess;
mem_cov_pre = blkdiag(initial_guess_center_cov, initial_guess_alpha_cov, ...
    initial_guess_len_cov, initial_guess_velo_cov);
mem_est = struct('state', []);
mem_est(1).state = mem_x_pre;
mem_est(1).cov = mem_cov_pre;
%state_dim = numel(mem_x_pre);

%% get Jacobians and Hessians
[f_func_g, f_jacobian_mat, f_hessian_mat] = get_jacobian_hessian(motionmodel, h1_var, h2_var);

figure;
hold on

for t = 1:time_steps
    %% get measurements
    meas_per_frame = poissrnd(possion_lambda);
    while meas_per_frame == 0
        meas_per_frame = poissrnd(possion_lambda);
    end
    disp(['time step:' num2str(t) ', ' num2str(meas_per_frame) ' measurements']);
    
    % generate measurements
    meas = zeros(2, meas_per_frame);
    for n = 1:meas_per_frame
        multi_noise(n, :) = -1 + 2.*rand(1, 2);
        while norm(multi_noise(n, :)) > 1
            multi_noise(n, :) = -1 + 2.*rand(1, 2);
        end
        
        meas(:, n) = gt(1:2, t) + multi_noise(n, 1)*gt(4, t)*...
            [cos(gt(3, t)); sin(gt(3, t))] + multi_noise(n, 2)*gt(5, t)*...
            [-sin(gt(3, t)); cos(gt(3, t))] + mvnrnd([0 0], meas_noise(:, :, t), 1)';
    end
    
    %% measurement update
    for n = 1:meas_per_frame
        
        [mem_x, mem_cov] = measurement_update(mem_x_pre, mem_cov_pre, meas(:, n), ...
            f_func_g, f_jacobian_mat, f_hessian_mat, meas_noise(:, :, t), multi_noise_cov);
        
        mem_x_pre = mem_x;
        mem_cov_pre = mem_cov;
    end
    theta_est = mem_x(3);
    mem_center = mem_x(1:2);
    mem_length = mem_x(4:5);
    mem_rot_mat = [cos(theta_est) -sin(theta_est); sin(theta_est) cos(theta_est)];
    
    
    mem_est(t).state = mem_x;
    mem_est(t).cov = mem_cov;
    %% time update
    mem_cov = process_matrix*mem_cov*process_matrix' + mem_process_noise;
    mem_x = process_matrix*mem_x;
    mem_x_pre = mem_x;
    mem_cov_pre = mem_cov;
    
    %% visualize estimate and ground truth for every 3rd scan
    if mod(t, 3)==1
        meas_points=plot( meas(1, :), meas(2, :), '.k', 'lineWidth', 0.5);
        hold on
        axis equal
        gt_plot = plot_extent(gt(1:2, t), gt(4:5, t), '-', 'k', 1, gt_rotation(:, :, t));
        est_plot = plot_extent(mem_center, mem_length, '-', 'g', 1, mem_rot_mat);
        pause(0.1)
    end
end
xlabel('x');
ylabel('y');
legend([gt_plot, est_plot, meas_points], {'Ground truth', 'Estimate', 'Measurement'});
