%% Initialization
clear
close all
clc


% Setup the variables and constants
z = linspace(0, 70, 1000) * 0.3048; % [m]
d_b = 20*0.0254;
d_t = 8.7*0.0254;
t = 0.2*0.0254;
z_0 = 70*0.3048;

% calculate the moment of intertia
I = (pi*(d_b - (z*(d_b - d_t))/z_0).^4)/64 - (pi*(2*t - d_b + (z*(d_b - d_t))/z_0).^4)/64;

% Fit a polynomial to the data
p = polyfit(z, I, 2);
I_p = polyval(p, z);

% Plots
figure
hold on
plot(z, I)
plot(z, I_p)

