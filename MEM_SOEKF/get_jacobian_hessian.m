function [ f_func_g, f_jacobian, f_hessian] = get_jacobian_hessian(motionmodel, h1_var, h2_var)
% GET_JACOBIAN_HESSIAN: calculates the Jacobian and Hessians (using the Matlab symbolic toolbox)
% Input:
%       motionmodel:   string, either 'static' or 'NCV'
%       h1_var:     variance of multiplicatice error h1
%       h2_var:     variance of multiplicatice error h2
% Output:
%       f_func_g:         quadratic function handle
%       f_jacobian_mat:   handle of Jacobian matrix of func_g with modified
%                         substitution
%       f_hessian_mat:    handle of Hessian matrix of funct_g with modified
%                         substitution

syms m1 m2 h1 h2 l1 l2 v1 v2 s1 s2 a

func_g =[(m1 + h1*l1*cos(a) - h2*l2*sin(a) + v1);...
    (m2 + h1*l1*sin(a) + h2*l2*cos(a) + v2);...
    (m1 + h1*l1*cos(a) - h2*l2*sin(a) + v1)^2;...
    (m2 + h1*l1*sin(a) + h2*l2*cos(a) + v2)^2;...
    (m1 + h1*l1*cos(a) - h2*l2*sin(a) + v1)*(m2 + h1*l1*sin(a) + h2*l2*cos(a) + v2)];

if strcmp(motionmodel, 'static')
    X = [m1;m2; a;l1;l2;h1;h2;v1;v2];
    
elseif strcmp(motionmodel, 'NCV')
    X = [m1;m2; a;l1;l2;s1;s2;h1;h2;v1;v2];
    
else
    error('unknown dynamic model');
end

jacobian_mat = jacobian(func_g, X);

for i = 1:numel(func_g)
    hessian_mat(:, :, i) = hessian(func_g(i), X);
end
jacobian_mat = subs(expand(jacobian_mat), [h1^2, h2^2], [h1_var, h2_var]);
hessian_mat = subs(expand(hessian_mat), [h1^2, h2^2], [h1_var,  h2_var]);

f_func_g = matlabFunction(func_g, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});
f_jacobian = matlabFunction(jacobian_mat, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});
f_hessian = matlabFunction(hessian_mat, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});

end