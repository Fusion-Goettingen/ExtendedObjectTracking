function [ f_g, f_jac, f_hes] = get_jacobian_hessian(motionmodel, C_h)
% GET_JACOBIAN_HESSIAN: calculates the Jacobian and Hessians (using the Matlab symbolic toolbox)
% Input:
%       motionmodel:   string, either 'static' or 'NCV'
%       h1_var:     variance of multiplicatice error h1
%       h2_var:     variance of multiplicatice error h2
% Output:
%       f_g:         quadratic function handle
%       f_jac: handle of Jacobian matrix of func_g with modified substitution
%       f_hes: handle of Hessian matrix of funct_g with modified substitution

syms m1 m2 h1 h2 l1 l2 v1 v2 s1 s2 a
y1 = (m1 + h1*l1*cos(a) - h2*l2*sin(a) + v1);
y2 = (m2 + h1*l1*sin(a) + h2*l2*cos(a) + v2);

func_g =[y1;y2;y1^2;y1*y2;y2^2];
h1_var = C_h(1,1);
h2_var = C_h(2,2);
if strcmp(motionmodel, 'static')
    X = [m1;m2; a;l1;l2;h1;h2;v1;v2];
    
elseif strcmp(motionmodel, 'NCV')
    X = [m1;m2;s1;s2; a;l1;l2;h1;h2;v1;v2];
    
else
    error('unknown dynamic model');
end

jacobian_mat = jacobian(func_g, X);

for i = 1:numel(func_g)
    hessian_mat(:, :, i) = hessian(func_g(i), X);
end
jacobian_mat = subs(expand(jacobian_mat), [h1^2, h2^2], [h1_var, h2_var]);
hessian_mat = subs(expand(hessian_mat), [h1^2, h2^2], [h1_var,  h2_var]);

f_g = matlabFunction(func_g, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});
f_jac = matlabFunction(jacobian_mat, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});
f_hes = matlabFunction(hessian_mat, 'Vars', {[m1, m2, a, l1, l2, h1, h2, v1, v2]});

end