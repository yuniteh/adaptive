function [AR] = arAlgo(dataWin,arOrder)
%# codegen
%%%%%%%%%%%%%%%%%%%%%% 
% August 25, 2013
%Levi Hargrove
%
%Inputs:
%dataWin: single channel of data from which you want to extract AR features
%arOrder:  order of the AR model  
%
%Outputs:
%AR: vector of AR coefficients.  Ignore the first coefficient as it will always be 1
%%%%%%%%%

AR = zeros(arOrder+1,1);
AR(1) = 1;
K = zeros(arOrder+1,1); 
R0 = dot(dataWin,dataWin);
R = zeros(1,arOrder);
for i = 1:arOrder
   R(i) = dot(dataWin(1:end-i),dataWin(i+1:end)); 
end
E = R0;
AR(2) = -R(1)/R0;

K(1) = AR(2);
q = R(1);
tmp = zeros(1,arOrder);

for i = 1:arOrder-1
    
    E = E + q*K(i);
    q = R(i+1);
    S = 0;
    for k = 1:i
        disp(R(k))
        disp(AR(i+2-k))
        S = S + R(k)*AR(i+2-k);
    end
    q = q+S;
    K(i+1)= -q/E;
    for k = 1:i
        tmp(k) = K(i+1)*AR(i+2-k);
    end
    for k = 2:i+1  
        AR(k) = AR(k) + tmp(k-1);
    end
    AR(i+2) = K(i+1);
end

