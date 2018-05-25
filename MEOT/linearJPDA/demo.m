% Implementation of the multiple extended object tracking algorithm based on the article
% 
% "Linear-Time Joint Probabilistic Data Association for Multiple Extended Object Tracking (to appear)"
% S. Yang, K. Thormann, and M. Baum
% 2018 IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM 2018), Sheffield, United Kingdom, 2018.
% 
%
%
% Further information:
% http://www.fusion.informatik.uni-goettingen.de
% https://github.com/Fusion-Goettingen
% 
% Source code written by Shishan Yang
% =============================

clc
close all
clear
dbstop error


scenario = 1; % turn and closely-spaced
% scenario = 2; % cross
clambda = 40; % clutter rate


% motion and measurement parameters used for multiplicative noise model
Cv = diag([10 10]);
Crw = diag([10 10 10 10]);
Cpw(:,:,1) = diag([0.02 1 1]);
Cpw(:,:,2) = diag([0.02 1 1]);
Ar = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
Ap = eye(3);
H = [eye(2);zeros(2,2)]';



figure
hold on
box on
axis equal

[gt, meas,mlambda, xbound, ybound,cp,nr_timesteps] = getMeasGt(scenario,clambda,Cv);


% first guess 
r = [-20 -250 10 10;-20 250 10 -10];
p = [0 30 30;0 15 15 ];

N = size(r,1);

Cr(:,:,1) = diag([900 900 10 10]);
Cp(:,:,1) = diag([.2 400 400]);

Cr(:,:,2) = diag([900 900 10 10]);
Cp(:,:,2) = diag([.02 100 100]);


%% plot first guess
for n = 1:N
    plot_extent([r(n,1:2) p(n,:)],'--','r',1);
end


for t = 1:nr_timesteps
    
    [r,p,Cr,Cp] = MEOT_JPDA(meas{t},r,p,Cr,Cp,cp,H,Cv,mlambda,clambda);
    
    %% Visulize  
    if mod(t,3)==1
        
        pMeas = plot(meas{t}(1,:),meas{t}(2,:),'k.');
        for n = 1:N
            plotGT = plot_extent(gt(n,1:5,t),'-','k',1);            
            plotEst = plot_extent([r(n,1:2),p(n,:)],'-','g',1);
        end        
        pause(0.001)
    end

    
    %% prediction
    for n = 1:N
        r(n,:) = Ar*r(n,:)';
        p(n,:) = Ap*p(n,:)';
        Cr(:,:,n) = Ar*Cr(:,:,n)*Ar'+Crw;
        Cp(:,:,n) = Ap*Cp(:,:,n)*Ap'+Cpw(:,:,n);
    end
end
    legend([plotGT plotEst],{'Ground Truth','Estimates'})
