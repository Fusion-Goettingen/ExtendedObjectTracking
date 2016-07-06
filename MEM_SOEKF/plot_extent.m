function [handle_extent] = plot_extent(c, l, line_style, color, line_width, rotation)
% PLOT_EXTENT: to plot the elliptical extent
% Input: 
%           c:              object center 2x1
%           l:              semi-axis length 2x1
%           line_style:     defined the same as in Matlab plot function
%           color:          defined the same as in Matlab plot function
%           line_width:     defined the same as in Matlab plot function
%           roation:        ration matrix, 2x2
% Output: 
%           handle_extent: the handle of the object extent

alpha = 0:pi/100:2*pi;

points = diag(l)*[cos(alpha); sin(alpha)];

rotated_points = rotation* points;
shifted_points = rotated_points + repmat(c, 1, size(points, 2));

handle_extent = plot(shifted_points(1, :),shifted_points(2,:),'LineStyle',...
    line_style,'color',color,'LineWidth',line_width);
end