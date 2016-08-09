function [points1, points2] = get_uniform_points_boundary(ellipse1, ellipse2, nr_points)
% GET_UNIFORM_POINTS_BOUNDARY gives the sets of points that are
% equidistantly chosen on the boudary of two ellipses, so that future OSPA 
% could be calculated
%
% Input:
%        ellipse1, 1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
%        ellipse2, 1x5, parameterization of the other ellispe [m1 m2 alpha l1 l2]
%        nr_points, nr of points that are uniformly chosen on the boundary
%                   to calculate OPSA distance
% Output:
%       points1, 2xnr_points, points that are chosen on the ellipse1
%       poitns2, 2xnr_points, points that are chosen on the ellispe2
% Written by Shishan Yang

theta = (0:2*pi/nr_points:2*pi-2*pi/nr_points);

alpha1 = ellipse1(3);
center1 = ellipse1(1:2)';
l1 = ellipse1(4:5);
gt_rotation_mat = [cos(alpha1) -sin(alpha1); sin(alpha1) cos(alpha1)];
points1(1,:) = l1(1)*cos(theta);
points1(2,:) = l1(2)*sin(theta);

alpha2 = ellipse2(3);
center2 = ellipse2(1:2)';
l2 = ellipse2(4:5);
est_rotation_mat = [cos(alpha2) -sin(alpha2); sin(alpha2) cos(alpha2)];
points2(1,:) = l2(1)*cos(theta);
points2(2,:) = l2(2)*sin(theta);



points1 = gt_rotation_mat*points1 + repmat(center1,1,size(points1,2));
points2 = est_rotation_mat*points2 + repmat(center2,1,size(points2,2));
end