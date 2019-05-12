%% init
clear
close all
clc

%% create some frequency data
F_s = 128;
N_fft = 1024;
N_b = 255;

f_y = 10.1;

%% Generate the signals
time = (0:1/F_s:(N_fft-1)/F_s)';
y = sin(f_y*2*pi*time) + 1e-2*randn(size(time));

%% Calculate the frequency transform
[freq, mag] = my_fft(time, y, 'Window', 'None');
[freq_window, mag_window] = my_fft(time, y);

%% Plots
figure
hold on
plot(freq, mag, 'k', 'DisplayName', 'Standard FFT (N=1024)')
plot(freq_window, mag_window, 'b', 'DisplayName', 'Windowed FFT, Blackman (N=1024)')
set(gca, 'Yscale', 'log')
grid on
xlabel('Frequency [Hz]')
ylabel('Signal Strength')
title('FFT Windowing example')
legend('show')


function [freq, fft_out] = my_fft(time, signal, varargin)
	% my_fft - This function calculates the DFT (discrete Fourier transform) of a signal.  This uses
	% MATLAB's native fft() function.
	%
	% Author: Ryan Takatsuka
	% Last Revision: 05-May-2019
	%
	% Syntax:
	%	[freq, fft_out] = my_fft(time, signal)
	%	[freq, fft_out] = my_fft(time, signal, 'Window', 'Blackman', 'Output', 'rms')
	%   
	% Inputs:
	%	time (vector): The time vector [seconds]
	%	signal (vector): The signal vector to be transformed to the frequency domain
	%
	% Outputs:
	%	freq (vector): Frequency vector [Hz]
	%	fft_out (vector): The DFT output that is in the specified output form
	%
	% Examples: 
	%	[freq, mag] = my_fft(time, signal);
	%	plot(freq, mag)
	% 	set(gca, 'Yscale', 'log')
	%
	%	This example calculates the FFT of the given signal and plots the response with a
	%	logarithmic scale for the magnitude vector.
	%
	% Notes:
	%	This function uses a windowing filter to calculate the DC components, which is more accurate
	%	than a standard FFT, but slower.
	%
	% Dependencies:
	%	fft, hamming, blackman
	
	% Input parsing
	p = inputParser;
	p.addOptional('Window', 'Blackman')
	p.addOptional('Output', 'rms')
	parse(p, varargin{:})
	
	if length(signal)/2 ~= floor(length(signal)/2)
		signal = signal(1:end-1);
	end
	
	% Design the window function
	if strcmp(p.Results.Window, 'None')
		w = ones(size(signal));
	elseif strcmp(p.Results.Window, 'Hamming')
		w = hamming(length(signal));
	elseif strcmp(p.Results.Window, 'Blackman')
		w = blackman(length(signal));
	else
		error('Not a valid window type!')
	end
	
	% Reshape the inputs
	signal = reshape(signal, [length(signal),1]);
	
	% Calculate the length of the signal
	N = length(signal);
	
	% Calculate the magnitude vector
	% FFT of the signal vector (magnitude of real and imaginary components
	mag = abs(fft(signal.*w));
	% Convert to a single-sided vector and fix factor of 2 scaling
	mag = mag(1:N/2+1);
	mag(2:end-1) = 2*mag(2:end-1);
	
	% Calculate the sampling frequency of the data
	Fs = 1/(mean(diff(time)));
	
	% Calculate the frequency vector for the data
	freq = transpose(Fs * (0:N/2) / N);
	freq_bin = freq(end) - freq(end-1);
	
	% Calculate window gain compensation
	K = sum(w);
	
	% Format the magnitude data in the output type
	if strcmp(p.Results.Output, 'rms')
		fft_out = mag / K;
	elseif strcmp(p.Results.Output, 'rms_db')
		fft_out = mag2db(mag / K);
	elseif strcmp(p.Results.Output, 'power')
		fft_out = mag.^2 / K;
	elseif strcmp(p.Results.Output, 'power_db')
		fft_out = pow2db(mag.^2 / K);
	elseif strcmp(p.Results.Output, 'power_density_db')
		fft_out = pow2db(mag.^2 / K / freq_bin);
	else
		error('Not a valid output type!')
	end

end