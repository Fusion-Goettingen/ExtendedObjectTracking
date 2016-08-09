% This script shows the different assignments for uniform OSPA distances
% when four points and 50 points are chosen on the boundary (Fig.4) of our paper:
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

nr_points_boundary = 50; % nr of points that used for the calculation of the uniform OSPA


% ellipse parameterization: [center1,center2,angle,length 0f
% semmi-axis1,length of semmi-axis2]
% 
%% set the ground truth
gt = [0 0 0 1 2];

%% set the estimate
est = [0 0 pi/6 1 2];


figure
hold on
gt_plot = plot_extent(gt, '-', 'k', 1);

est_plot = plot_extent(est,'--','g',1);
axis equal
xlim([-2,2])
ylim([-2,2])
grid on
box on


[gt_points_4, est_points_4] = get_uniform_points_boundary(gt, est,4);
[gt_points_m, est_points_m] = get_uniform_points_boundary(gt,est,nr_points_boundary);



[~, assignment_m]= ospa_dist(gt_points_m,est_points_m,10000,2);

for j = 1:size(gt_points_m,2)
    [~,b(j)]=find(assignment_m(j,:));
    plot([gt_points_m(1,j) est_points_m(1,b(j))],[gt_points_m(2,j) est_points_m(2,b(j))],'r--');
end


[~, assignment_4]= ospa_dist(gt_points_4,est_points_4,10000,2);

for j = 1:size(gt_points_4,2)
    [~,b(j)]=find(assignment_4(j,:));
    gt_points = plot(gt_points_4(1,j),gt_points_4(2,j),'k*','LineWidth',5);
    est_points = plot(est_points_4(1,j),est_points_4(2,j),'g*','LineWidth',5);
    plot([gt_points_4(1,j) est_points_4(1,b(j))],[gt_points_4(2,j) est_points_4(2,b(j))],'k');
end
legend([gt_plot,est_plot],{'Ground Truth','Estimate'})
