function [ x_est, x_cov ] = measurement_update(mem_x_pre, mem_cov_pre, meas, ...
    f_func_g, f_jacobian, f_hessian, meas_noise_cov, multi_noise_cov)

% MEASUREMENT_UPDATE: estimate state using second order extended kalman filter (SOEKF)
% Input:
%       mem_x_pre:      previous estimate, [m1, m2, alpha, l1, l2, (velo1, velo2)]
%                       5x1 for 'static', 7x1 for 'NCV'
%       mem_cov_pre:    previous covariance matrix 
%                       5x5 for 'static', 7x7 for 'NCV'
%       meas:           measurement, 2x1 
%       f_func_g:         quadratic function handle, output of GET_JACOBIAN_HESSIAN function
%       f_jacobian:       handle of Jacobian of func_g, output of GET_JACOBIAN_HESSIAN function
%       f_hessian:        handle of Hessians of func_g, output of GET_JACOBIAN_HESSIAN function
%       meas_noise_cov: covariance of measurement noise, 2x2
%       multi_noise_cov: covariance of multiplicative noise, 2x2, diag(h1_var, h2_var)
% Output:
%       x_est: estimated state
%       x_cov: covariance of state estimate

dim_state = numel(mem_x_pre);
nr_param = 5; % 5 paramtets: m1, m2, alpha, l1, l2  



%% shift center to improve robustness
shifted = mem_x_pre(1:2);
meas = meas - shifted;
mem_x_pre_shifted = mem_x_pre;
mem_x_pre_shifted(1:2)=[0;0];

%% augment state and its covariance 
X_est = [mem_x_pre_shifted;0;0;0;0];
X_cov = blkdiag(mem_cov_pre, multi_noise_cov, meas_noise_cov);

%% construct pseudo-measurement
z = [meas(1);meas(2);meas(1)^2;meas(2)^2; meas(1)*meas(2)];
 
%% Substitute quadratic function, Jacobian and Hessian matrices using current estimate
val_subs = [mem_x_pre_shifted(1:nr_param);0;0;0;0]; % velocity does not appear in measurement equation

subs_func_g = f_func_g(val_subs');
subs_jacobian = f_jacobian(val_subs');
subs_hessian = f_hessian(val_subs');

%% calculate variance of pseudo-measurement 
cov_zz = zeros(nr_param, nr_param);
for i = 1:nr_param
    for j = 1:nr_param
        cov_zz(i, j) = subs_jacobian(i, :)*X_cov*subs_jacobian(j, :)'+ ...
            (1/2)*trace(subs_hessian(:, :, i)*X_cov*subs_hessian(:, :, j)*X_cov);
        
    end
end

%% calculate mean of pseudo-measurement 
mean_z = zeros(1, nr_param);
for i = 1:nr_param
    mean_z(i) = subs_func_g(i) + (1/2)*trace(subs_hessian(:, :, i)*X_cov);
end

%% calculate cross-variance of pseudo-measurement with augmented estimate
cov_Xz = X_cov*subs_jacobian';

%% Kalman filter update
X_est = double(X_est + cov_Xz*cov_zz^-1 *( z - mean_z'));
X_cov = double(X_cov - cov_Xz*cov_zz^-1*cov_Xz');

X_est(1:2) = X_est(1:2) + shifted;

%% Truncate augmented estimate and covariance
        X_cov = (X_cov+X_cov')/2; % enforce covariance matrix symmetric
        x_est = X_est(1:dim_state); 
        x_cov = X_cov(1:dim_state, 1:dim_state);
end