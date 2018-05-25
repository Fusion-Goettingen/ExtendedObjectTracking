% Implementation of the multiple extended object tracking algorithm based on the article
%
% "Linear-Time Joint Probabilistic Data Association for Multiple Extended Object Tracking (to appear)"
% S. Yang, K. Thormann, and M. Baum
% 2018 IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM 2018), Sheffield, United Kingdom, 2018.
%
%
% Further information:
% http://www.fusion.informatik.uni-goettingen.de
% https://github.com/Fusion-Goettingen
%
% Source code written by Shishan Yang
% =============================
function [r,p,Cr,Cp] = MEOT_JPDA(meas,r,p,Cr,Cp,cp,H,Cv,mlambda,clambda)

M = size(meas,2);
N = size(r,1);
G = zeros(N,M);


% calculating Eq.(4)
for n = 1:N
    % for further information, we refer you to
    % "Tracking the Orientation and Axes Lengths of an Elliptical Extended Object"
    % Shishan Yang and Marcus Baum
    % arXiv preprint, 2018
    % https://arxiv.org/abs/1805.03276
    Cy(:,:,n) = get_pred_meas_cov(p(n,:),Cr(:,:,n),Cp(:,:,n),H,Cv);
    
    for m = 1:M
        if (meas(:,m)'-r(n,1:2))*Cy(:,:,n)^-1*(meas(:,m)-r(n,1:2)') < 16
            G(n,m) = mvnpdf(meas(:,m),r(n,1:2)',Cy(:,:,n));
        end
    end
end

% get the mariginal association probability
beta = mariginalAssocationProb(G,cp,mlambda,clambda);

for n = 1:N
    beta_n = beta(n,:);
    matched_ind = beta_n>0;
    beta_n = beta_n(matched_ind);
    obs_n = meas(:,matched_ind);
    
    [r(n,:),p(n,:),Cr(:,:,n),Cp(:,:,n)] = prob_measurement_update(H,r(n,:)',p(n,:)',Cr(:,:,n),Cp(:,:,n),Cv,obs_n,beta_n);
    
end
end


function beta = mariginalAssocationProb(G,cp,mlambda,clambda)
% get the mariginal assocation probability accroding Eq (15)
[N,M]=size(G);
beta = zeros(N+1,M);
betaTemp = zeros(N+1,M);

for j = 1:M
    S(j) = clambda*cp;
    for n = 1:N
        S(j) = S(j) + mlambda(n)*G(n,j);
    end
end
for j = 1:M
    for i = 1:N
        
        betaTemp(i,j) = mlambda(i)*G(i,j)/S(j);
    end
    betaTemp(N+1,j) =  clambda*cp/S(j);
end

% normalization
for j = 1:M
    beta(:,j) = betaTemp(:,j)./sum(betaTemp(:,j));
end
end

function P = get_pred_meas_cov(p,Cr,Cp,H,R)
% get predicted measurement covariance
%
% for further information, we refer you to
% "Tracking the Orientation and Axes Lengths of an Elliptical Extended Object"
% Shishan Yang and Marcus Baum
% arXiv preprint, 2018
% https://arxiv.org/abs/1805.03276

Ch = diag([.25 .25]);
alpha = p(1);
l1 = p(2);
l2 = p(3);

S =[cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]*diag([l1 l2]);
J(:,:,1) = [-l1*sin(alpha) cos(alpha) 0; -l2*cos(alpha) 0 -sin(alpha)];
J(:,:,2) = [l1*cos(alpha) sin(alpha) 0; -l2*sin(alpha) 0 cos(alpha)];


for m = 1:2
    for n = 1:2
        CII(m,n) = trace(Cp*J(:,:,n)'*Ch*J(:,:,m));
    end
end
CI = S*Ch*S';
P = H*Cr*H' + CI + CII + R;

end