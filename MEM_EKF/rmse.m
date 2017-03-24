% This script compares three elliptical extended object trackig methods.
% It gives Fig.3 in
% S. Yang and M. Baum Extended Kalman Filter for Extended Object Tracking.
% Proceedings of the 2017 IEEE International Conference on Acoustics, Speech,
% and Signal Processing (ICASSP), New Orleans, USA, 2017.


close all
clc
clear
dbstop warning
set(0,'defaulttextinterpreter','latex')

%% parameters

mc_runs = 500;
% nr of measurements we get follows a possion distribution
possion_lambda = 5;
H = [1 0 0 0; 0 1 0 0]; % matrix maps kinematic state into position
H_velo = [zeros(2,2),eye(2)];
H_par_SOEKF = [eye(2),zeros(2,5);zeros(3,4),eye(3,3)];
H_velo_SOEKF =  [zeros(2,2),eye(2),zeros(2,3)];
motionmodel = {'NCV'};

%% generate ground truth
[gt_kin,gt_par, time_steps, delta_t] =get_ground_truth;

d_RMM=zeros(mc_runs,time_steps);
d_SOEKF=zeros(mc_runs,time_steps);
d_EKF=zeros(mc_runs,time_steps);

v_RMM=zeros(mc_runs,time_steps);
v_SOEKF=zeros(mc_runs,time_steps);
v_EKF=zeros(mc_runs,time_steps);
%% setting prior
hat_r0 = [100,100,5,-8]'; % kinematic state: position and velocity
hat_p0 = [-pi/3,200,90]'; % shape variable: orientation and semi-axes lengths
hat_x0 = [hat_r0; hat_p0];

C_r0 = blkdiag( 900*eye(2),16*eye(2));
C_p0 = blkdiag(0.02*eye(1),400*eye(2));
C_x0 = blkdiag(C_r0,C_p0);


% parameters for EKF
C_h = diag([1/4, 1/4]);
C_v = 0.2*diag([100^2,20^2]);

Ar = [eye(2),delta_t*eye(2); zeros(2,2),eye(2)];
Ap = eye(3);

C_w_r = blkdiag(100*eye(2),eye(2)); % process noise covariance for kinematic state
C_w_p = blkdiag(0.04,0.5*eye(2)); % process noise covariance for shape variable

for it = 1:mc_runs
    
hat_r_EKF = [100,100,5,-8]'; % kinematic state: position and velocity
hat_p_EKF = [-pi/3,200,90]';
Cr_EKF = C_r0;
Cp_EKF = C_p0;

% parameters for SOEKF
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

    
        %% evaluation
        
        d_RMM(it,t) = d_gaussian_wasserstein(gt_cur_par,rmm_par);
        d_SOEKF(it,t) = d_gaussian_wasserstein(gt_cur_par,H_par_SOEKF*hat_x_SOEKF);
        d_EKF(it,t) = d_gaussian_wasserstein(gt_cur_par,[H*hat_r_EKF; hat_p_EKF]);
        
        v_RMM(it,t) = norm(gt_velo - H_velo*hat_x_RMM);
        v_SOEKF(it,t)=norm(gt_velo - H_velo_SOEKF*hat_x_SOEKF);
        v_EKF(it,t) = norm(gt_velo - H_velo*hat_r_EKF);
        
        
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
end
figure;
hold on
plot(1:time_steps,mean(d_RMM),'g',1:time_steps,mean(d_SOEKF),'r',1:time_steps,mean(d_EKF),'b')
legend({'Random Matrices','SOEKF','EKF'})
title('Extension RMSE')
box on
grid on

figure;
hold on
plot(1:time_steps,mean(v_RMM),'g',1:time_steps,mean(v_SOEKF),'r',1:time_steps,mean(v_EKF),'b')
legend({'Random Matrices','SOEKF','EKF'})
title('Velocity RMSE')
box on
grid on
