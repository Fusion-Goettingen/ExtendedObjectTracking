function [ hat_x,Cx]= predictSOEKF(A,hat_x,C_x,C_w)    
    
    Cx = A*C_x*A' + C_w;
    hat_x = A*hat_x;
end