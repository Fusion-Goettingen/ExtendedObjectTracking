% This script gives the scenario 2 and its error plot (2nd column of Fig.5) of our paper:
% Shishan Yang, Marcus Baum, and Karl Granstroem. "Metrics for Performance
% Evaluation of Ellipitical Extended Object Tracking Methods",
% The 2016 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI 2016)
% ---BibTeX entry
% @InProceedings{MFI16_Yang,
%   Title                    = {{Metrics for Performance Evaluation of Elliptic Extended Object Tracking Methods}},
%   Author                   = {Shishan Yang and Marcus Baum and Karl Granstr\"om},
%   Booktitle                = {{IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI 2016)}},
%   Year                     = {2016},
%
%   Owner                    = {yang},
%   Timestamp                = {2016.07.01}
% }


clc
close all
clear
dbstop error

addpath('hungarian/')

nr_points_boundary = 100;
nr_bins = 100;


% ellipse parameterization: [center1,center2,angle,length 0f
% semmi-axis1,length of semmi-axis2]
%
%% set the ground truth
gt_m = [0 0]';
gt_alpha = 0;
gt_l1 = 1;
gt_l2 = 2;
gt = [gt_m; gt_alpha; gt_l1; gt_l2]';

%% set the estimate
est_m =[0 0]';
est_alpha = 5*pi/6;
est_l1 = 1;
est_l2 = linspace(0.1,2*gt_l2,nr_bins)';
est = [repmat(est_m',nr_bins,1), est_alpha*ones(nr_bins,1), est_l1*ones(nr_bins,1), est_l2];


%% initialization
d_gw= zeros(nr_bins,1);
d_kl= zeros(nr_bins,1);
d_uniform_OSPA4= zeros(nr_bins,1);
d_uniform_OSPA100 = zeros(nr_bins,1);
centers_err = zeros(nr_bins,1);


%% visuliztion
figure
subplot(1,2,1)
gt_plot = plot_extent(gt, '-', 'k',1);
axis equal
xlim([-4,4])
ylim([-4,4])
grid on
box on


subplot(1,2,2)
xlim([0,2])
ylim([0,3])
grid on
box on


for i = 1:nr_bins
    
    if i>1
        delete(est_plot)
    end
    
    len_scale = est_l2/gt_l2;
    [gt_points_4, est_points_4] = get_uniform_points_boundary(gt, est(i,:),4);
    [gt_points_m, est_points_m] = get_uniform_points_boundary(gt,est(i,:),nr_points_boundary);
    
    d_gw(i) = d_gaussian_wasserstein(gt, est(i,:));
    d_kl(i) = d_kullback_leibler(gt', est(i,:)');
    
    % the cut-off value in OSPA is setted as a very big value:1000000
    [d_uniform_OSPA4(i), ~]= ospa_dist(gt_points_4, est_points_4,1000000,2);
    [d_uniform_OSPA100(i),~]= ospa_dist(gt_points_m,est_points_m,1000000,2);
    
    
    %% visulization
    subplot(1,2,1)
    hold on
    est_plot = plot_extent(est(i,:), '--', 'g', 1);
    legend([gt_plot,est_plot],{'Ground Truth','Estimate'});
    
    subplot(1,2,2)
    hold on
    plot(len_scale(i), d_kl(i),'b+', len_scale(i), d_gw(i), 'r*',...
        len_scale(i), d_uniform_OSPA4(i),'go', len_scale(i),d_uniform_OSPA100(i),'k.')
    error_leg=legend({'$d_{KL}$','$d_{GW}$','$d_{OSPA_4}$','$d_{OSPA_{100}}$'});
    set(error_leg,'Interpreter','latex');
    pause(0.000001)
end


