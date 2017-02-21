function [rmm_rotation, rmm_l, fai] = ...
    get_random_matrix_ellipse(rmm_extent)
% notations and formmulas are from "Ellipse fitting based approach for extended object tracking"
% Equation (9)
rho = (rmm_extent(1,1) - rmm_extent(2,2))/(2*rmm_extent(1,2));

fai = atan(-rho + sqrt(1 + rho^2));

rmm_rotation = [cos(fai), -sin(fai); sin(fai), cos(fai)];
rmm_l_sq = diag(rmm_rotation'*rmm_extent*rmm_rotation);
rmm_l = rmm_l_sq.^(1/2);
end