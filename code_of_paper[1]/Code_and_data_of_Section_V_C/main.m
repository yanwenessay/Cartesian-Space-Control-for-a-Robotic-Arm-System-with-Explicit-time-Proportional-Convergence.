function [u,z1,z2]   = fcn(x1,x2,M,V,G,F,thetad_ddot)

%M,V,G,F The identified dynamic block
%thetad_ddot Joint angular velocity expectation
%x1 Joint angular position error, x2 Joint angular velocity error


x1=x1(1,1);
x2=x2(1,1);
M=M(1,1);
thetad_ddot=thetad_ddot(1,1);

forwardfeed=(V'+G+F');

Tc=1.5;
z1s=0.5*pi/180;
z2s=0.5*pi/180;
% x1c=0.1744; %10^o
% x2c=xx; %30^o
z1c=0.1744;
z2c=1;
f=0.001;

z1=x1;

a1=-log(z1c/z1s)/(Tc)*z1-f^2/(2*z2s^2)*z1;

a1dot=-log(z1c/z1s)/(Tc)*x2-f^2/(2*z2s^2)*x2;

z2=x2-a1;

ucom=-z2/2-log(z2c/z2s)/Tc*z2-z2s^2/z1s^2*z1+a1dot+thetad_ddot;
u=(M+15)*ucom+forwardfeed(1,1)-(1/((abs(z1)/1)^20+0.1))*atan(z2/0.003);


end
