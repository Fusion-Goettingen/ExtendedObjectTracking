function [x_update, X_update,P_update, alpha_update] = updateRMM(....
    x, X, P, alpha,y,y_cov,R,n_k,H,const_z)
% formulas are from :
% M. Feldmann and D. Fraenken. "Tracking of Extended
% Objects and Group Targets using Random Matrices - A Performance
% Analysis"

% by Shishan Yang
X_sqrt = sqrtm(X);
Y = const_z * X + R;
Y_sqrt_inv = (sqrtm(Y))^(-1);

S = H * P * H' + Y/n_k;
S_sqrt = sqrtm(S);
% the bold_dot is: inverse_sqrt_S*(y-H*x)
bold_dot =  S_sqrt\(y - H * x);
N_hat = (X_sqrt * bold_dot) * (X_sqrt * bold_dot)';


K = P * H' * S^(-1);

alpha_update = alpha + n_k;
P_update = P - K*S*K';
x_update = x + K*(y-H*x);
X_update = (1/alpha_update)*(alpha*X + N_hat + (X_sqrt*Y_sqrt_inv)*y_cov*(X_sqrt*Y_sqrt_inv)');



end