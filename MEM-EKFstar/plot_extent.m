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
% =============================

function [handle_extent] = plot_extent(ellipse,line_style, color, line_width)
% PLOT_EXTENT plots the extent of an ellipse or circle
% Input:
%        ellipse1,    1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
%        line_style,  defined as in the Matlab plot function
%        color,       defined as in the Matlab plot function
%        line_width,  defined as in the Matlab plot function
%
% Output:
%        handle_extent, the handle of the plot
%
% Written by Shishan Yang



center = ellipse(1:2);
theta = ellipse(3);
l = ellipse(4:5);
R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix


alpha = 0:pi/100:2*pi;
xunit = l(1)*cos(alpha);
yunit = l(2)*sin(alpha);

rotated = R* [xunit; yunit];
xpoints = rotated(1,:) + center(1);
ypoints = rotated(2,:) + center(2);


handle_extent = plot(xpoints,ypoints,'LineStyle',line_style,'color',color,'LineWidth',line_width);

end