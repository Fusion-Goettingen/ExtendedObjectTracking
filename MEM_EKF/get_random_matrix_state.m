function A = get_random_matrix_state(ellipse_extent)
% only ellipse parameterization alpha is implemented 
% by Shishan Yang
alpha = ellipse_extent(1);
eigen_val = ellipse_extent(2:3);
eigen_vec = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
A = eigen_vec*diag(eigen_val.^2)*eigen_vec';
A = (A+A')/2; % make A symmetric
end