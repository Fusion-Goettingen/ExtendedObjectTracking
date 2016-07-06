function [gt_center, gt_rotation, gt_orient, gt_length, gt_vel, time_steps, time_interval] = get_ground_truth

%% -------------Setting ground truth---------------------------------------
gt_center(:, 1) = [0, 0]';
% trajectory
gt_orient = [ repmat(-pi/4, 1, 20), (-pi/4: pi/40:0), ...
   zeros(1, 10), (0: pi/20:2*pi/4), ....
    repmat(2*pi/4, 1, 20), (2*pi/4: pi/20:pi), ....
    repmat(pi, 1, 20)];
% assume object is aligned along its velocity
gt_vel = [(500/36)*cos(gt_orient);(500/36)*sin(gt_orient)];
gt_length = repmat([340/2;80/2], 1, size(gt_vel, 2));

time_steps = size(gt_vel, 2);
time_interval = 10;


gt_rotation = zeros(2, 2, time_steps);
for t = 1 : time_steps
    gt_rotation(:, :, t) = [cos(gt_orient(t)), -sin(gt_orient(t)); sin(gt_orient(t)), cos(gt_orient(t))];
    if t>1
        gt_center(:, t) = gt_center(:, t-1) + gt_vel(:, t)*time_interval;
    end
end


end

