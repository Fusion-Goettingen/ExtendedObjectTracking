function [kld] = d_kullback_leibler( ellipse1, ellipse2)
% D_KULLBACK_LEIBLER gives the Kullback-Leibler Divergence between two
% ellipses which could interpreted as Gaussians
%
% Input:
%        ellipse1, 1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
%        ellipse2, 1x5, parameterization of the other ellispe [m1 m2 alpha l1 l2]
%
% Output:
%        kld, scalar, Kullback-Leibler divergence
%
% Written by Shishan Yang


m1 = ellipse1(1:2);
dim = numel(m1);


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

kld = .5*(trace((sigma1^(-1))*sigma2) + (m1 - m2)'*(sigma1^(-1))*(m1 - m2)...
    - dim + log((det(sigma1))/det(sigma2)) );


end

