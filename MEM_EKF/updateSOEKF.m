function [ hat_x, C_x ] = updateSOEKF(hat_x, C_x, y, ...
    f_g, f_jac, f_hes, C_v, C_h)

% MEASUREMENT_UPDATE: estimate state using second order extended kalman filter (SOEKF)
% Input:
%       hat_x: previous estimate, [m1, m2,(velo1, velo2), alpha, l1, l2], 5x1 for 'static', 7x1 for 'NCV'
%       C_x:   previous covariance matrix, 5x5 for 'static', 7x7 for 'NCV'
%       y:     measurement, 2x1
%       f_g:   quadratic function handle, output of GET_JACOBIAN_HESSIAN function
%       f_jac: handle of Jacobian of func_g, output of GET_JACOBIAN_HESSIAN function
%       f_hes: handle of Hessians of func_g, output of GET_JACOBIAN_HESSIAN function
%       C_v:   covariance of measurement noise, 2x2
%       C_h:   covariance of multiplicative noise, 2x2, diag(h1_var, h2_var)
% Output:
%       hat_x: updated state
%       C_x:   updated covariance 

dim_state = numel(hat_x);
nr_param = 5; %


%% shift center to improve robustness
shifted = hat_x(1:2);
y = y - shifted;
mem_x_pre_shifted = hat_x;
mem_x_pre_shifted(1:2)=[0;0];

%% augment state and its covariance
X_est = [mem_x_pre_shifted;0;0;0;0];
X_cov = blkdiag(C_x, C_h, C_v);

%% construct pseudo-measurement
y1 = y(1);
y2 = y(2);
Y = [y1;y2;y1^2;y1*y2;y2^2];

%% Substitute quadratic function, Jacobian and Hessian matrices using current estimate
val_subs = [0;0;mem_x_pre_shifted(5:7);0;0;0;0]; % velocity does not appear in measurement equation

hat_g = f_g(val_subs');
hat_jacobian = f_jac(val_subs');
hat_hessian = f_hes(val_subs');

%% calculate variance of pseudo-measurement
cov_YY = zeros(nr_param, nr_param);
for i = 1:nr_param
    for j = 1:nr_param
        cov_YY(i, j) = hat_jacobian(i, :)*X_cov*hat_jacobian(j, :)'+ ...
            (1/2)*trace(hat_hessian(:, :, i)*X_cov*hat_hessian(:, :, j)*X_cov);
        
    end
end

%% calculate mean of pseudo-measurement
E_Y = zeros(1, nr_param);
for i = 1:nr_param
    E_Y(i) = hat_g(i) + (1/2)*trace(hat_hessian(:, :, i)*X_cov);
end


%% calculate cross-variance of pseudo-measurement with augmented estimate
cov_Xz = X_cov*hat_jacobian';

%% Kalman filter update
X_est = double(X_est + cov_Xz*cov_YY^-1 *( Y - E_Y'));
X_cov = double(X_cov - cov_Xz*cov_YY^-1*cov_Xz');

X_est(1:2) = X_est(1:2) + shifted;

%% Truncate augmented estimate and covariance
X_cov = (X_cov+X_cov')/2; % enforce covariance matrix symmetric
hat_x = X_est(1:dim_state);
C_x = X_cov(1:dim_state, 1:dim_state);
end