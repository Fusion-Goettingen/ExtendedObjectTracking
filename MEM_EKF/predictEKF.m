function [hat_r, C_r, hat_p, C_p] = predictEKF(Ar,Ap, hat_r, hat_p, C_r, C_p, C_w_r, C_w_p)
    hat_r = Ar*hat_r;
    hat_p = Ap*hat_p;
    C_r = Ar*C_r*Ar' + C_w_r;
    C_p = Ap*C_p*Ap' + C_w_p;
end