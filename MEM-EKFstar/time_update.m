% Implementation of the MEM-EKF* algorithm based on the article
% 
% "Tracking the Orientation and Axes Lengths of an Elliptical Extended Object"
% Shishan Yang and Marcus Baum
% arXiv preprint, 2018,
% https://arxiv.org/abs/1805.03276
% 
% Further information:
% http://www.fusion.informatik.uni-goettingen.de
% https://github.com/Fusion-Goettingen
% 
% Source code written by Shishan Yang
% =============================

function [r,p,Cr,Cp]= time_update(r,p,Cr,Cp,Ar, Ap,Cwr, Cwp)
r = Ar*r;
Cr = Ar*Cr*Ar'+Cwr;

p = Ap*p;
Cp = Ap*Cp*Ap'+Cwp;
end