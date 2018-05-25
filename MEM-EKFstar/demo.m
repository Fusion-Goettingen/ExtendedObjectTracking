% Implementation of the MEM-EKF* algorithm based on the article
% 
% "Tracking the Orientation and Axes Lengths of an Elliptical Extended Object"
% Shishan Yang and Marcus Baum
% arXiv preprint, 2018,
% https://arxiv.org/abs/1805.03276
% 
% Further information:
% http://www.fusion.informatik.uni-goettingen.de
% https://github.com/Fusion-Goettingen
% 
% Source code written by Shishan Yang

close all
clc
clear
dbstop error

% generate ground truth
[gt_center, gt_rotation, gt_orient, gt_length, gt_vel, time_steps, time_interval] =get_ground_truth;
gt = [gt_center;gt_orient;gt_length;gt_vel];

% nearly constant velocity model 
H = [1 0 0 0; 0 1 0 0];
Ar =[1 0 10 0; 0 1 0 10; 0 0 1 0; 0 0 0 1] ;
Ap = eye(3);

Ch = diag([1/4, 1/4]); % covariance of the multiplicative noise
Cv = diag([200 8]); % covariance of the measurement noise
Cwr = diag([100 100 1 1]); % covariance of the process noise for the kinematic state
Cwp = diag([0.05 0.001 0.001]); %covariance of the process noise for the shape parameters

lambda = 5;% Nr of measurements is Poisson distributed with mean lambda


%% Prior
r = [100, 100,10, -17]';
p = [-pi/3 200 90]';

Cr = diag([900 900 16 16]);
Cp = diag([0.2 400 400]);

figure;
hold on
for t = 1:time_steps
    %% generate measurements
    nk = poissrnd(lambda);
    while nk == 0
        nk = poissrnd(lambda);
    end
    disp(['Time step: ' num2str(t) ', ' num2str(nk) ' Measurements']);
    
    y = zeros(2, nk);
    for n = 1:nk
        h(n, :) = -1 + 2.*rand(1, 2);
        while norm(h(n, :)) > 1
            h(n, :) = -1 + 2.*rand(1, 2);
        end
        
        y(:, n) = gt(1:2, t) + h(n, 1)*gt(4, t)*...
            [cos(gt(3, t)); sin(gt(3, t))] + h(n, 2)*gt(5, t)*...
            [-sin(gt(3, t)); cos(gt(3, t))] + mvnrnd([0 0], Cv, 1)';
    end
    
    %% measurement update
    [r,p,Cr,Cp] = measurement_update(y,H,r,p,Cr,Cp,Ch,Cv);
    
    
    %% visualize estimate and ground truth for every 3rd scan
    if mod(t, 3)==1
        meas_points=plot( y(1, :), y(2, :), '.k', 'lineWidth', 0.5);
        hold on
        axis equal
        gt_plot = plot_extent(gt(:, t), '-', 'k', 1);
        est_plot = plot_extent([r(1:2);p ], '-', 'r', 1);
        pause(0.1)
    end
    
    %% time update
    [r,p,Cr,Cp]= time_update(r,p,Cr,Cp,Ar,Ap,Cwr,Cwp);
    
    
end
legend([gt_plot, est_plot, meas_points], {'Ground truth', 'Estimate', 'Measurement'},'Location','northwest');
