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
function [gt, meas,mlambda, xbound, ybound,cp,nr_timesteps] = getMeasGt(scenario,clambda,R)
rot = @ (a) [cos(a) -sin(a); sin(a) cos(a)];
    gt(:,:,1) = [0 -300 -pi/3 40 15; 0 300 pi/4 20 10];
    mlambda = [9 7];
if scenario == 1

    velo_p1t1 = repmat([11;7.7],1,30);
    velo_p1t2 = repmat([11;-7.7],1,30);
    
    dy1 = velo_p1t1(end):-.8:0;
    dy2 = velo_p1t2(end):.8:0;
    
    da1 = gt(1,3,1):(-pi/2-gt(1,3,1))/(numel(dy1)-1):-pi/2;
    da2 = gt(2,3,1):((pi/2)-gt(2,3,1))/(numel(dy2)-1):pi/2;
    
    velo_p2t1 =  [repmat(velo_p1t1(1),1,numel(dy1)); dy1];
    velo_p2t2 =  [repmat(velo_p1t2(1),1,numel(dy1)); dy2];
    
    velo_p3t1 = repmat([11;0],1,40);
    velo_p3t2 = repmat([11;0],1,40);
    
    
    velo(:,:,1) = [velo_p1t1 velo_p2t1 velo_p3t1];
    velo(:,:,2) = [velo_p1t2 velo_p2t2 velo_p3t2];
    
    alpha(1,:) = [repmat(gt(1,3,1),1,size(velo_p1t1,2)) da1 repmat(da1(end),1,size(velo_p3t1,2))];
    alpha(2,:) = [repmat(gt(2,3,1),1,size(velo_p1t2,2)) da2 repmat(da2(end),1,size(velo_p3t2,2))];
   
   gt(1,6:7,1)=velo(:,1,1);
   gt(2,6:7,1)=velo(:,1,2);
elseif scenario == 2

  

    velo(:,:,1) = repmat([10;10],1,60);

    velo(:,:,2) = repmat([10;-10],1,60);
    alpha(1,:) = repmat(gt(1,3,1),1,size(velo,2));
    alpha(2,:) = repmat(gt(2,3,1),1,size(velo,2));
   gt(1,6:7,1)=velo(:,1,1);
   gt(2,6:7,1)=velo(:,1,2);
end


temp=[];
for i = 1:size(gt,1)
    nr_meas = poissrnd(mlambda(i));
    for j = 1:nr_meas
        h(:,j) = mvnrnd([0;0],diag([.25 .25]));
        temp = [temp gt(i,1:2,1)' + rot(gt(i,3,1))*diag(gt(i,4:5,1))*h(:,j)+mvnrnd([0;0],R)'];
    end
    if ~isempty(temp)
    
   
    xmax(1) = max(temp(1,:));
    xmin(1) = min(temp(1,:));
    ymax(1) = max(temp(2,:));
    ymin(1) = min(temp(2,:));
    end
end
meas{1}=temp;


nr_timesteps = size(velo,2);

for t = 2:nr_timesteps
    temp = [];

    
    for i = 1:size(gt,1)
        if i~=1 || t< nr_timesteps%-20 % target 1 is terminated
        gt(i,1:2,t) = gt(i,1:2,t-1)+velo(:,t,i)';
        gt(i,3,t)=alpha(i,t);
        gt(i,4:5,t) = gt(i,4:5,t-1);
        gt(i,6:7,t) = velo(:,t,i)';
        nr_meas = poissrnd(mlambda(i));
        for j = 1:nr_meas
            h(:,j) = mvnrnd([0;0],diag([.25 .25]));
            temp = [temp gt(i,1:2,t)' + rot(gt(i,3,t))*diag(gt(i,4:5,t))*h(:,j)+mvnrnd([0;0],R)'];
        end
        
        end

        if ~isempty(temp)
            xmax(t) = max(temp(1,:));
            xmin(t) = min(temp(1,:));
            ymax(t) = max(temp(2,:));
            ymin(t) = min(temp(2,:));
        end
    end
    meas{t}=temp;
    
end
xmax = round(max(xmax));
xmin = floor(min(xmin));
xbound = [xmin xmax];

ymax = round(max(ymax));
ymin = floor(min(ymin));
ybound = [ymin ymax];
xlim(xbound)
ylim(ybound)

for t= 1:nr_timesteps
    nr_clutter =  poissrnd(clambda);
    clutter{t}(1,:) = (rand(nr_clutter,1)'* (xmax-xmin)) + xmin;
    clutter{t}(2,:) = (rand(nr_clutter,1)' * (ymax-ymin)) + ymin;
    
    cp = 1 / ((xmax-xmin)*(ymax-ymin));
    meas{t} = [meas{t} clutter{t}];
end

end