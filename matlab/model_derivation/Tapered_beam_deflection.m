%% Tapered_beam_deflection.m
% Calculate the deflection for a tapered beam

clear all
close all
clc

%% Define symbolic variables
syms F L E I1 I2 I x M v theta

%% Moment equation
M =  F*x - F*L;

%% Moment of Inertia
I = (I2 - I1)/L *x + I1;

%% Curvature equation
v = M/E/I;

%% Slope Equation
theta = int(v, x);

%% Deflection Equation