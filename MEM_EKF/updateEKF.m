function [ r_hat, C_r,p_hat, C_p ] = updateEKF(r_hat, C_r, p_hat, C_p, y, C_v, C_h)

[S_hat,M_hat] = getMS(p_hat,C_h);


H = [1 0 0 0;0 1 0 0];

% moments of kinematic state
E_y = H*r_hat;
C_ry = C_r*H';
C_yy = H*C_r*H' +  S_hat*C_h*S_hat'+C_v;

% kinematic state update
r_hat = r_hat + C_ry*(C_yy)^(-1)*(y-E_y);

C_r = C_r - C_ry*(C_yy)^(-1)*C_ry';
C_r = (C_r+C_r')/2;

%% pseudo measurement
y_shift = y - E_y; % shift the measuremet to match central moments
% Take the 2nd kronecker product of shifted measurements
% and delete the duplicated element we get the pseudo measurement
Y = [eye(2),zeros(2,2);0 0 0 1]*kron(y_shift,y_shift); 
% moments of pseudo-measurement
sgm11 = C_yy(1,1);
sgm12 = C_yy(1,2);
sgm22 = C_yy(2,2);
E_Y = [sgm11; sgm12; sgm22];
C_YY = [3*sgm11^2, 3*sgm11*sgm12,sgm11*sgm22 + 2*sgm12^2 ;...
        3*sgm11*sgm12, sgm11*sgm22 + 2*sgm12^2, 3*sgm22*sgm12;
       sgm11*sgm22 + 2*sgm12^2, 3*sgm22*sgm12,3*sgm22^2];
C_pY = C_p*M_hat'; 
% shape variable update
p_hat = p_hat + C_pY*(C_YY)^(-1)*(Y-E_Y);
C_p = C_p - C_pY*(C_YY)^(-1)*C_pY';
C_p = (C_p + C_p')/2;
end

function [S, M] = getMS(p,C_h)
a = p(1);
l1 = p(2);
l2 = p(3);
S = [cos(a) -sin(a); sin(a) cos(a)]*diag([l1 l2]);
M = [-sin(2*a), (cos(a))^2, (sin(a))^2  ;...
      cos(2*a), sin(2*a),   -sin(2*a);...     
      sin(2*a), (sin(a))^2, (cos(a))^2]*...
     diag([l1^2*C_h(1,1)-l2^2*C_h(2,2),2*l1*C_h(1,1),2*l2*C_h(2,2)]);
end