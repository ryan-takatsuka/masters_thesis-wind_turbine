%% numerical_ky.m
% Solve for ky numerically

%% Initialization
clear
close all
clc

%% Constants
z_0 = 70 * u.ft;
d_t = 8.7 * u.in;
d_b = 20 * u.in;
t = 0.2 * u.in;
E = 30e6 * u.psi;
F = 1 * u.N;
m = 474 * u.kg;

z = linspace(0*u.ft, z_0, 1000);

% Convert to non dimensional units
% z_0 = u2num(z_0);
% x_0 = u2num(x_0);
% d_t = u2num(d_t);
% t_b = u2num(d_b);
% t = u2num(t);
% E = u2num(E);
% z = u2num(z);

%% Equations
% Moment
M = (z - z_0);

% Diameter
d = d_b - z*(d_b-d_t)/z_0;
% d = d_t * ones(size(z));

% Moment of inertia
I = pi/64*d.^4 - (pi/64)*(d-2*t).^4;

% Curvature
v = -M ./ (E*I);

% Slope (theta)
theta = cumtrapz(z, v);

% deflection
y = cumtrapz(z, theta);

% Spring constant
ky = 1 ./ y;


%% Extract values from data and remove units
z = z ./ u.m;
I = I ./ u.m^4;
theta = theta ./ u.s^2 .* u.m .*u.kg;
v = v ./ u.s^2 .* u.m^2  .* u.kg;
y = y ./ u.m .* u.N;

%% Plots
figure
subplot(2,2,1)
plot(z, I, 'k', 'LineWidth', 2)
title('Moment of interia')
xlabel('Height, z [m]')
ylabel('I [m^4]')
grid on

subplot(2,2,2)
plot(z, v, 'k', 'LineWidth', 2)
title('Curvature / Force')
xlabel('Height, z [m]')
ylabel('v/F [s^2/m^2/kg]')
grid on

subplot(2,2,3)
plot(z, theta, 'k', 'LineWidth', 2)
title('Slope / Force')
xlabel('Height, z [m]')
ylabel('\theta/F [s^2/m/kg]')
grid on

subplot(2,2,4)
plot(z, y, 'k', 'LineWidth', 2)
title('Displacement / Force')
xlabel('Height, z [m]')
ylabel('y/F [m/N]')
grid on



ky(end)

wn = sqrt(ky(end) / m)


%% tests


