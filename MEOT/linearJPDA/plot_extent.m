% Implementation of the multiple extended object tracking algorithm based on the article
% 
% "Linear-Time Joint Probabilistic Data Association for Multiple Extended Object Tracking (to appear)"
% S. Yang, K. Thormann, and M. Baum
% 2018 IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM 2018), Sheffield, United Kingdom, 2018.
% 
%
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
%        line_style,  definedthe same as in Matlab plot function
%        color,       definedthe same as in Matlab plot function
%        line_width,  definedthe same as in Matlab plot function
%
% Output:
%        handle_extent, the handle of the plot
%
% Written by Shishan Yang



center = ellipse(1:2);
theta = ellipse(3);
l = ellipse(4:5);
R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix


alpha = 0:pi/30:2*pi;
xunit = l(1)*cos(alpha);
yunit = l(2)*sin(alpha);

rotated = R* [xunit; yunit];
xpoints = rotated(1,:) + center(1);
ypoints = rotated(2,:) + center(2);


handle_extent = plot(xpoints,ypoints,'LineStyle',line_style,'color',color,'LineWidth',line_width);

end