%% numerical_m_eff.m
% Solve for m_eff numerically

%% Initialization
clear
close all
clc

%% Constants
z_0 = 70 * u.ft; % The height of the tower
d_t = 6 * u.in; % The diameter of the top of the tower 8.7
d_b = 13.957 * u.in; % The diameter of the base of the tower 20
t = 0.375 * u.in; % The thickness of the tower 0.2
E = 29.2e6 * u.psi; % The modulus of elasticity
F = 1 * u.N; % The normalized force
m_nacelle = 474 * u.lb; % mass of the nacelle
rho0 = 8050 * u.kg/(u.m^3); % density of the tower
rho_weight = 0.284 * u.lbf / u.in^3; % weight density
rho = rho_weight / (32 * u.ft/u.s^2); % mass density

% Create the vector for the tower height
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

% Calculate the cross sectional area
A = (pi*(d/2).^2) - (pi*((d-2*t)/2).^2);

% Calculate the normalized displacement
y_n = y / max(abs(y));

% Calculate the effective mass at the end of the tower
m_eff = cumtrapz(z, rho*A.*(y_n.^2));

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

figure
plot(z, m_eff ./ u.kg, 'k', 'LineWidth', 2)
title('Effective mass')
xlabel('Height, z [m]')
ylabel('Mass [kg]')
grid on


% Calculate the total mass
m_total = m_nacelle + m_eff(end)
m_tower = trapz(z, A)*rho

% Print out the spring constant 
ky(end)

% Calculate the natural frequency
wn = sqrt(ky(end) / m_total);
fn = wn/2/pi


