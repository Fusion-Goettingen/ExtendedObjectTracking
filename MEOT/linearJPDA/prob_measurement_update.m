% Implementation of the multiple extended object tracking algorithm based on the article
% 
% "Linear-Time Joint Probabilistic Data Association for Multiple Extended Object Tracking (to appear)"
% S. Yang, K. Thormann, and M. Baum
% 2018 IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM 2018), Sheffield, United Kingdom, 2018.
% 
% The extended object kinematic and shape state update formulas see
% "Tracking the Orientation and Axes Lengths of an Elliptical Extended Object"
% Shishan Yang and Marcus Baum
% arXiv preprint, 2018
% https://arxiv.org/abs/1805.03276
%
%
% Further information:
% http://www.fusion.informatik.uni-goettingen.de
% https://github.com/Fusion-Goettingen
% 
% Source code written by Shishan Yang
% =============================
function [r,p,Cr,Cp] = prob_measurement_update(H,r,p,Cr,Cp,Cv,y,beta_j)

nk = size(y,2);  % number of measurements at time k
Ch = diag([.25 .25]);




for i = 1:nk
    
    [CI,CII,M,F,Ftilde] = get_auxiliary_variables(p,Cp,Ch);
    
    yi = y(:,i);
    py = beta_j(i);
    
    
    % calculate moments for the kinematic state update
    y_bar = H*r;
    Cry = Cr*H';
    Cy = H*Cr*H' + CI + CII + Cv;
    
    % udpate kinematic estimate
    Kr = Cry*Cy^-1;
    
    r = r + Kr*(py* (yi-y_bar));
    Cr =  Cr-py*(Kr*H*Cr) + py^3*(1-py)*Kr*( (yi-y_bar)* (yi-y_bar)')*Kr';
    % Enforce symmetry of the covariance
    Cr = (Cr+Cr')/2;
    
    
    % construct pseudo-measurement for the shape update
    Yi = F*kron( (yi-y_bar), (yi-y_bar));
    
    Yibar = F * reshape(Cy,4,1);
    CpY = Cp*M';
    CYY = F*kron(Cy,Cy)*(F+Ftilde)';
    
    % update shape
    Kp= CpY*CYY^(-1);
    
    p = p + Kp*py*(Yi-Yibar);
    Cp = Cp-py*(CpY*CYY^(-1)*CpY') + py^3*(1-py)*Kp*((Yi-Yibar)*(Yi-Yibar)')*Kp';
    % Enforce symmetry of the covariance
    Cp = (Cp+Cp')/2;
    
end

end

function [CI,CII,M,F,Ftilde] = get_auxiliary_variables(p,Cp,Ch)
alpha = p(1);
l1 = p(2);
l2 = p(3);

S = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]*diag([l1 l2]);
S1 = S(1,:);
S2 = S(2,:);

J1 = [-l1*sin(alpha) cos(alpha) 0; -l2*cos(alpha) 0 -sin(alpha)];
J2 = [ l1*cos(alpha) sin(alpha) 0; -l2*sin(alpha) 0  cos(alpha)];

CI = S*Ch*S';
CII(1,1) = trace(Cp*J1'*Ch*J1);
CII(1,2) = trace(Cp*J2'*Ch*J1);
CII(2,1) = trace(Cp*J1'*Ch*J2);
CII(2,2) = trace(Cp*J2'*Ch*J2);

M = [2*S1*Ch*J1; 2*S2*Ch*J2; S1*Ch*J2 + S2*Ch*J1];

F = [1 0 0 0; 0 0 0 1; 0 1 0 0];
Ftilde = [1 0 0 0; 0 0 0 1; 0 0 1 0];
end