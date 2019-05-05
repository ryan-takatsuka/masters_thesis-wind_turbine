clear
close all
clc

syms z d_b d_t t z_0

z = linspace(0, 70, 1000) * 0.3048; % [m]
d_b = 20*0.0254;
d_t = 8.7*0.0254;
t = 0.2*0.0254;
z_0 = 70*0.3048;


I = (pi*(d_b - (z*(d_b - d_t))/z_0).^4)/64 - (pi*(2*t - d_b + (z*(d_b - d_t))/z_0).^4)/64;


p = polyfit(z, I, 2);
I_p = polyval(p, z);


figure
hold on
plot(z, I)
plot(z, I_p)

