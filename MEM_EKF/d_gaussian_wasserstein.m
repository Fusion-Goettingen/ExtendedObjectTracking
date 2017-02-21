function [distance] = d_gaussian_wasserstein( ellipse1, ellipse2)
% Gaussian wasserstein distance between two ellipses
%
% Reference:
% S. Yang, M. Baum, and K. Granström. Metrics for Performance Evaluation of
% of Elliptic Extended Object Tracking Methods. Proceedings of the 2016 
% IEEE International Conference on Multisensor Fusion and Integration for 
% Intelligent Systems (MFI 2016), Baden-Baden, Germany, 2016.
% 
% Input:
%        ellipse1, 1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
%        ellipse2, 1x5, parameterization of the other ellispe [m1 m2 alpha l1 l2]
%
% Output:
%       distance, scalar, gaussian wasserstein distance
%
% Written by Shishan Yang

m1 = ellipse1(1:2);
alpha1 = ellipse1(3);
eigen_val1 = [ellipse1(4), ellipse1(5)];
eigen_vec1 = [cos(alpha1), -sin(alpha1); sin(alpha1), cos(alpha1)];
sigma1 = eigen_vec1*diag(eigen_val1.^2)*eigen_vec1';
sigma1 = (sigma1 + sigma1')/2; % make covariance symmetric



m2 = ellipse2(1:2);
alpha2 = ellipse2(3);
eigen_val2 = [ellipse2(4),ellipse2(5)];
eigen_vec2 = [cos(alpha2), -sin(alpha2); sin(alpha2), cos(alpha2)];
sigma2 = eigen_vec2*diag(eigen_val2.^2)*eigen_vec2';
sigma2 = (sigma2 + sigma2')/2; % make covariance symmetric

error_sq = norm(m1-m2)^2 + trace(sigma1 + sigma2 -2*sqrtm((sqrtm(sigma1)*sigma2*sqrtm(sigma1))));
distance = sqrt(error_sq);


end