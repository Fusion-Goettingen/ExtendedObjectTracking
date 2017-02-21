function [x_pred, X_pred,P_pred, alpha_pred] = predictRMM(....
    x, X, P, alpha,F,Q,T,tau)
% formulas are from :
% M. Feldmann and D. Fraenken. "Tracking of Extended
% Objects and Group Targets using Random Matrices - A Pereformance
% Analysis"

% by Shishan Yang

x_pred = F * x;
P_pred = F * P * F' + Q;

X_pred = X;
alpha_pred = 2 + exp(-T/tau)*(alpha - 2);

end