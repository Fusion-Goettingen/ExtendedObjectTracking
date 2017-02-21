function [ x_est, x_cov ] = measurement_update_ekf(mem_x_pre, mem_cov_pre, meas, ...
    f_Jh, f_Jp,  meas_noise_cov, multi_noise_cov)




est_mm = [mem_x_pre(1:2);mem_x_pre(6:7)];
cov_mm = [mem_cov_pre(1:2,1:2) mem_cov_pre(1:2,6:7);mem_cov_pre(6:7,1:2) mem_cov_pre(6:7,6:7)];
cov_pp = mem_cov_pre(3:5,3:5);
Jm = [eye(2) zeros(2,2)];

subs_Jh = f_Jh(mem_x_pre(3:5)');
subs_Jp = f_Jp(mem_x_pre(3:5)');
%% construct pseudo-measurement

Y= [(meas(1) - mem_x_pre(1))^2;...
    (meas(2) - mem_x_pre(2))^2;...
    (meas(1) - mem_x_pre(1))*(meas(2) - mem_x_pre(2))];
%% kinematic update
cov_yy = Jm*cov_mm*Jm' + subs_Jh*multi_noise_cov*subs_Jh' + meas_noise_cov;
cov_my = cov_mm*Jm';

est_kin =  est_mm + cov_my* cov_yy^-1* (meas-Jm*est_mm(1:4));
kin_cov =  cov_mm -  cov_my* cov_yy^-1*cov_my';
kin_cov = (kin_cov+kin_cov')/2;
%% shape update

sg11 = cov_yy(1,1);
sg12 = cov_yy(1,2);
sg22 = cov_yy(2,2);

expect_Y = [sg11;sg22;sg12];

cov_xY = cov_pp*subs_Jp';
cov_YY = [3*sg11^2, sg11*sg22+ 2*sg12^2, 3*sg11*sg12;...
      sg11*sg22+ 2*sg12^2,3*sg22^2 , 3*sg22*sg12;...
      3*sg11*sg12, 3*sg22*sg12,sg11*sg22+ 2*sg12^2];


  
  
      
est_shape =  mem_x_pre(3:5) + cov_xY* cov_YY^-1* (Y- expect_Y);
shape_cov =  cov_pp - cov_xY* cov_YY^-1*cov_xY';
  
x_est = [est_kin(1:2);est_shape;est_kin(3:4)];
x_cov = [kin_cov(1:2,1:2), zeros(2,3), kin_cov(1:2,3:4);...
        zeros(3,2), shape_cov,zeros(3,2); ...
        kin_cov(3:4,1:2), zeros(2,3),kin_cov(3:4,3:4)];

end