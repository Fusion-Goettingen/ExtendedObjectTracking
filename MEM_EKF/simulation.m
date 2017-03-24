% This script compares three elliptical extended object trackig methods.
% It gives Fig.2 in
% S. Yang and M. Baum Extended Kalman Filter for Extended Object Tracking.
% Proceedings of the 2017 IEEE International Conference on Acoustics, Speech,
% and Signal Processing (ICASSP), New Orleans, USA, 2017.


close all
clc
clear
dbstop warning
set(0,'defaulttextinterpreter','latex')

%% parameters
motionmodel = {'NCV'};
% nr of measurements we get follows a possion distribution
possion_lambda = 5;
H = [1 0 0 0; 0 1 0 0]; % matrix maps kinematic state into position
H_velo = [zeros(2,2),eye(2)];
% parameters for EKF
C_h = diag([1/4, 1/4]);
C_v = 0.2*diag([100^2,20^2]);


%% generate ground truth
[gt_kin,gt_par, time_steps, delta_t] =get_ground_truth;

%% setting prior
hat_r0 = [100,100,5,-8]'; % kinematic state: position and velocity
hat_p0 = [-pi/3,200,90]'; % shape variable: orientation and semi-axes lengths
hat_x0 = [hat_r0; hat_p0];

C_r0 = blkdiag( 900*eye(2),16*eye(2));
C_p0 = blkdiag(0.02*eye(1),400*eye(2));
C_x0 = blkdiag(C_r0,C_p0);



Ar = [eye(2),delta_t*eye(2); zeros(2,2),eye(2)];
Ap = eye(3);

C_w_r = blkdiag(100*eye(2),eye(2)); % process noise covariance for kinematic state
C_w_p = blkdiag(0.04,0.5*eye(2)); % process noise covariance for shape variable

hat_r_EKF = [100,100,5,-8]'; % kinematic state: position and velocity
hat_p_EKF = [-pi/3,200,90]';
Cr_EKF = C_r0;
Cp_EKF = C_p0;

% parameters for SOEKF
sgh1 = 1/4;
sgh2 = 1/4;
C_w = blkdiag(C_w_r, C_w_p);
Ax = blkdiag(Ar,Ap);

hat_x_SOEKF = hat_x0;
Cx_SOEKF = C_x0;

[ f_g_ekf2, f_jacobian_ekf2, f_hessian_ekf2] = get_jacobian_hessian(motionmodel,C_h);


% parameters for Random Matrix
alpha = 50;
tau = 10;
T = 10;
const_z = 1/4;
hat_x_RMM = hat_r0;
hat_X_RMM = get_random_matrix_state(hat_p0);
Cx_RMM = C_r0;

figure;

hold on

for t = 1:time_steps
          N = poissrnd(possion_lambda);
        while N == 0
            N = poissrnd(possion_lambda);
        end
        disp(['time step:' num2str(t) ', ' num2str(N) ' measurements']);
        
        %% ------------------get measurements------------------------------------
        gt_cur_par = gt_par(:,t);
        gt_velo = H_velo*gt_kin(:,t);
        gt_rot = [cos(gt_cur_par(3)), -sin(gt_cur_par(3)); sin(gt_cur_par(3)), cos(gt_cur_par(3))];
        gt_len = gt_cur_par(4:5);
        y = zeros(2,N);
        for n = 1:N
            h_noise(n,:) = -1 + 2.*rand(1,2);
            while norm(h_noise(n,:)) > 1
                h_noise(n,:) = -1 + 2.*rand(1,2);
            end
            y(:,n) = H*gt_kin(:,t) + gt_rot*diag(gt_len)*h_noise(n,:)'+ mvnrnd([0 0], C_v, 1)';
        end
    %% update RMM
    meas_mean = mean(y,2);
    meas_spread = (N - 1) * cov(y');
    [hat_x_RMM, hat_X_RMM, C_x_RMM, alpha_update]...
        = updateRMM(hat_x_RMM, hat_X_RMM, Cx_RMM, alpha,meas_mean, ...
        meas_spread, C_v,N,H,const_z);
    
    [~, len_RMM,ang_RMM] = get_random_matrix_ellipse(hat_X_RMM);  
    rmm_par = [H*hat_x_RMM; ang_RMM;len_RMM];
    %% update EKF and SOEKF
    for n = 1:N
        [hat_x_SOEKF, Cx_SOEKF] = updateSOEKF(hat_x_SOEKF, Cx_SOEKF, y(:,n),...
            f_g_ekf2, f_jacobian_ekf2, f_hessian_ekf2, C_v, C_h);
        [ hat_r_EKF, Cr_EKF,hat_p_EKF, Cp_EKF ] = updateEKF(hat_r_EKF, Cr_EKF, hat_p_EKF, Cp_EKF, y(:,n), C_v, C_h);
        
    end
    
    
    %% visulization udpated shapes
    if mod(t,3)==1
        meas_points=plot( y(1,:)/1000, y(2,:)/1000, '.k','lineWidth',0.5);
        hold on
        gt_plot = plot_extent([gt_cur_par(1:2)/1000; gt_cur_par(3);gt_cur_par(4:5)/1000], '-','k',1);
        axis equal
        
        est_plot_rmm = plot_extent([rmm_par(1:2)/1000;rmm_par(3);rmm_par(4:5)/1000],'-','g',1);
        est_plot_ekf2 = plot_extent([hat_x_SOEKF(1:2)/1000;hat_x_SOEKF(5);hat_x_SOEKF(6:7)/1000],'-', 'r', 1);
        est_plot_ekf = plot_extent([H*hat_r_EKF/1000;hat_p_EKF(1);hat_p_EKF(2:3)/1000],'-', 'b', 1);
    end
    
    
    %%  predict　RMM
    [hat_x_RMM, hat_X_RMM,Cx_RMM, alpha] = predictRMM(....
        hat_x_RMM, hat_X_RMM, C_x_RMM, alpha_update,Ar,C_w_r,T,tau);
    if alpha_update<=2
        error('alpha<2')
    end
    %% predict SOEKF
    [hat_x_SOEKF,Cx_SOEKF] = predictSOEKF(Ax,hat_x_SOEKF,Cx_SOEKF,C_w);
     %% predict EKF
    [hat_r_EKF,Cr_EKF, hat_p_EKF,Cp_EKF] = predictEKF(Ar,Ap, hat_r_EKF, hat_p_EKF, Cr_EKF, Cp_EKF,C_w_r, C_w_p);
    
end
legend([meas_points,gt_plot,est_plot_rmm, est_plot_ekf2,est_plot_ekf],...
    {'measurement','ground truth','random matrix','SOEKF','EKF'})
box on
grid on
ylim([-3200 1200]/1000)
xlim([-200 8000]/1000)
