#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@file: split_dft
    This file breaks up a LifeLine data file, which contains a long sequence of
    DFT/rotor speed sequences. 

Created on Tue Mar 20 18:28:56 2018

@author: jr
'''

import os
import sys
import argparse
from matplotlib import pyplot, rcParams, cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.signal import savgol_filter


# ------------------------------------------------------------------------------

def pct_diff (num0, num1):
    ''' Find the percentage difference between two numbers, referenced to the
    first one.
    @param num0 The first number and basis for the percentage
    @param num1 The second number to be compared to the first '''

    try:
        pct = ((num1 - num0) / num1) * 100.0
    except ZeroDivisionError:
        if abs (num1) < 0.001:
            pct = 0.0
        else:
            pct = 100.0
    return pct


# ==============================================================================

class DFT_data:
    ''' This class holds a set of data for a DFT which has been taken by the
    LifeLine box. The data includes frequencies and relative amplitudes of
    acceleration data, information about the rotor speed while the DFT was
    being taken, and the time (relative to other DFT's during a session) at
    which the DFT was measured. '''

    def __init__ (self, frequencies = [], amplitudes = [], 
                  rotor_min = 999.9, rotor_mean = 0.0, rotor_max = 0.0, 
                  time = 0.0, low_cut = 10):
        ''' Initialize a DFT object. If given, use the data for DFT, speed,
        and time; if not, create an empty DFT object which can be filled in
        later.
        @param frequencies A list of frequency data
        @param amplitudes A list of amplitude data corresponding to frequencies
        @param rotor_min The minimum rotor speed during data collection
        @param rotor_mean The average rotor speed during data collection
        @param rotor_max The maximum rotor speed during data collection
        @param time The time, in seconds, since the LifeLine was turned on 
        @param low_cut The number of low-frequency data points to ignore '''

        # Fill the data
        self.num_readings = len (frequencies)
        self.freqs = frequencies
        self.ampls = amplitudes
        self.rotor_min = rotor_min
        self.rotor_mean = rotor_mean
        self.rotor_max = rotor_max
        self.time = time
        self.low_cut = low_cut

        # Make sure the frequencies and amplitudes are the same length
        if len (self.freqs) != len (self.ampls):
            print ('DFT_data ERROR: Different lengths for freqs ({:d}) and'
                   ' ampls ({:d})'.format (self.freqs, self.ampls))
            self.freqs = []
            self.ampls = []


    def add_in (self, other):
        ''' Add another DFT object's data to this one. 
        @param other The other DFT object whose data will be added in here.
            If this one is empty, just copy the other one in here '''

        # If this one is empty, copy the other one's data in here
        if not self.freqs:
            for item in other.freqs:
                self.freqs.append (item)
            for item in other.ampls:
                self.ampls.append (item)
            self.rotor_min = other.rotor_min
            self.rotor_mean = other.rotor_mean
            self.rotor_max = other.rotor_max
            self.time = other.time
            self.low_cut = other.low_cut

        # OK, it's addition time.  Make sure the lengths of lists agree
        else:
            if len (other.freqs) != len (self.freqs):
                print ('ERROR:  self.freqs {:d}, other.freqs {:d} long'.format \
                       (len (self.freqs), len (other.freqs)))
            if len (other.ampls) != len (self.ampls):
                print ('ERROR:  self.ampls {:d}, other.ampls {:d} long'.format \
                       (len (self.ampls), len (other.ampls)))
            else:
                for index in range (len (self.freqs)):
                    if pct_diff (self.freqs[index], other.freqs[index]) > 1.0:
                        print ('ERROR: Freq. difference {:f} != {:f}'.format (
                               self.freqs[index], other.freqs[index]))
                for index in range (len (self.ampls)):
                    self.ampls[index] += other.ampls[index]
            if other.rotor_min < self.rotor_min:
                self.rotor_min = other.rotor_min
            if other.rotor_max > self.rotor_max:
                self.rotor_max = other.rotor_max
#            self.time = other.time
#            self.low_cut = other.low_cut

            # Put the means together by relative weighting
            self_N = len (self.freqs)
            other_N = len (other.freqs)
            total_N = self_N + other_N
            self.rotor_mean *= self_N / total_N
            self.rotor_mean += other.rotor_mean * other_N / total_N


    def save (self, file_name, dir_name = '.'):
        ''' Save the DFT object's data in a file. The file will make it easy
        to create a plot. 
        @param dir_name The name of the directory in which to save the data
        @param file_name A name for the file in which to write the data '''

        full_name = dir_name + '/' + file_name + '.csv'
        print ('Saving: ' + full_name)
        with open (full_name, 'rw') as da_file:
            da_file.write ('DFT at {:.0f} sec, Rotor Speed {:.1f} RPM '
                  ' (Range {:.1f} to {:.1f})'.format (self.time, 
                  self.rotor_mean, self.rotor_min, self.rotor_max))
            for freq in self.freqs:
                ampl = self.ampls[self.freqs.index (freq)]
                da_file.write ('{:.1f},{.1f}'.format (freq, ampl))


    def plot (self, show = False, file_name = None, dir_name = '.'):
        ''' Make a plot of the DFT object's data using PyPlot and show the plot
        on the screen or save it to a file as ordered. 
        @param show Whether to show the file on screen
        @param file_name The name of a file to which to save the plot, or 
            @c None if the plot shouldn't be saved to a file 
        @param dir_name The name of the directory where the plot file is to be
            saved, @c '.' for the current directory by default '''

        if not show and not file_name:
            print ('DFT_data warning: plot() called for no reason')
            return

        # Compute the estimated time of program start
#        starting_seconds = 49598
#        now_seconds = starting_seconds + self.time
#        the_hr = int (now_seconds / 3600.0)
#        the_min = int ((now_seconds % 3600) / 60.0)
#        the_sec = int (now_seconds % 60)

        pyplot.figure (figsize = (12.80, 9.60), dpi = 200)
        rcParams.update({'font.size': 22})
        pyplot.plot (self.freqs[self.low_cut:], self.ampls[self.low_cut:], 
                     linewidth = 3.0)
        pyplot.xlabel ('Frequency (Hz)')
        pyplot.ylabel ('Component Amplitude (g-sec)')
        pyplot.title ('Balanced Rotor at {:.1f} RPM '.format (self.rotor_mean)
#        pyplot.title ('LifeLine DFT at {:02d}:{:02d}:{:02d}, '
#                      'Rotor Speed {:.1f} RPM '.format (the_hr, the_min, 
#                                                      the_sec, self.rotor_mean)
#                      ' (Range {:.1f} to {:.1f})'.format (self.time, 
#                      self.rotor_mean, self.rotor_min, self.rotor_max)
                     )
        if show:
            pyplot.show ()
        if file_name:
            # figsize=(3.841, 7.195), dpi=100
            full_name = dir_name + '/' + file_name + '.png'
            print ('Saving plot: ' + full_name)
            pyplot.savefig (full_name, dpi = 100)

        pyplot.close ('all')


# ------------------------------------------------------------------------------

def split_up (a_file, dir_name, base_name = 'DFT_', plots = False):
    ''' Break up the DFT data in @c a_file into smaller files and put each 
    smaller file into the given directory. The file must have been opened and
    the directory must have been created before this function is called. 
    @param a_file The open file with lots of DFT's
    @param a_dir The name of the directory where smaller files will be put 
    @param plots Whether to make plots of the DFT's to be saved or shown '''

    rotor_sum = 0.0       # Sum of rotor speeds, for finding average
    rotor_N = 0           # Number of rotor speed readings in set
    rotor_max = 0.0       # Maximum rotor speed during DFT data set
    rotor_min = 999.9     # Minimum rotor speed during DFT data set
    freqs = []            # Frequency data for DFT
    ampls = []            # Amplitude data for DFT

    file_count = 0        # Make file names by adding a count to a base name

    # A place to store the database of all DFT's
    all_DFTs = []

    while True:
        a_line = a_file.readline ()

        # Empty lines show up at the end of the file
        if not a_line:
            print ('Empty line')
            break

        else:
            # Any carriage return at the end is no longer needed
            a_line = a_line.strip ()

            # If the line consisted of nothing but a carriage return, ignore it
            if not a_line:
                pass

            # If the line begins with *, it should be rotor RPM
            elif a_line[0] == '*':
                try:
                    speed = float (a_line[1:])
                except ValueError:
                    print ('Bad rotor speed line: {:s}'.format (a_line))
                else:
                    rotor_N += 1
                    rotor_sum += float (a_line[1:])
                    rotor_max = rotor_max if rotor_max > speed else speed
                    rotor_min = rotor_min if rotor_min < speed else speed
    
            # If the line begins with @, it's part of the DFT
            elif a_line[0] == '@':
                try:
                    num_strs = a_line[1:].split (',')
                    freq = float (num_strs[0])
                    ampl = float (num_strs[1])
                except (ValueError, IndexError) as oopsie:
                    print ('Bad DFT line: ' + a_line)
                else:
                    freqs.append (freq)
                    # Scaling:  Divide by 1024 bits/g * 1024 samples in FFT *
                    #           50 Hz sampling rate
                    ampls.append (ampl / 52428800)
    
            # If the line begins with %, it's the end-of-DFT line, as in
            # "% End DFT 5417.23"
            elif a_line[0] == '%':
#                print (file_count, end = ' ')
                try:
                    line_time = float (a_line[10:])
                except ValueError as whoops:
                    print ('Bad End DFT line: ' + a_line)
                    line_time = 0.0
                else:            # Time is good, so save it
                    pass
    
                try:
                    rotor_mean = rotor_sum / rotor_N
                except ZeroDivisionError:
                    rotor_mean = 0.0
    
                new_DFT = DFT_data (freqs, ampls, rotor_min, rotor_mean,
                                    rotor_max, line_time)

                all_DFTs.append (new_DFT)

                if plots:
                    new_DFT.plot (show = False, 
                        file_name = '{:s}{:03d}'.format (base_name, file_count),                           
                        dir_name = dir_name)
                file_count += 1
    
                # Reset totals for the next data set
                rotor_sum = 0.0
                rotor_N = 0
                rotor_max = 0.0
                rotor_min = 999.9
                freqs = []
                ampls = []
    
    return all_DFTs


# ==============================================================================

# # parser = argparse.ArgumentParser (description='Split a LifeLine data file into'
# #                               ' a bunch of individual files in a subdirectory')
# # parser.add_argument ('file_name', help='input file name', action = 'store')
# # parser.add_argument ('dir_name', help='directory for individual files')
# # args = parser.parse_args ()
# class args_test:
#     def __init__(self):
#         self.file_name = 0
#         self.dir_name = 0
# args = args_test

# args.file_name = 'full_data.txt'
# args.dir_name = 'test_data'


# #print ('Args: ' + args.file_name + ',' + args.dir_name)

# # # Create the directory into which to put the split files
# # try:
# #     os.makedirs (args.dir_name)
# # except FileExistsError:
# #     print ('OHNOES: Directory {:s} exists! Try another name.')
# #     sys.exit (0)

# # Open the big file which is to be split up, then split it into DFT data sets.
# try:
#     with open (args.file_name, mode = 'r') as big_file:
#         print ('File opened.')
#         DFT_list = split_up (big_file, args.dir_name, plots = False)
#         print ('Size of DFT list: {:d} items'.format (len (DFT_list)))

# except FileNotFoundError:
#     print ('File "' + args.file_name + '" does not exist.')

# else:
#     print ('Woot.')
#     # Here's where we can do interesting things with the list of DFT's
# #    DFT_sum = DFT_data ()
# #    for a_dft in DFT_list[224:229]:
# #        DFT_sum.add_in (a_dft)
# #    DFT_sum.plot (show = True)

# print ('Program finished.')


# # -----------------------------
# #    Start Post Processing
# # -----------------------------
# # The value to cutoff of the data
# low_cut = 25

# # Iterate through the DFTs and add the good data to some lists
# amplitudes = []
# time = []
# for DFT in DFT_list:
#     try:
#         if len(DFT.ampls)!=256:
#             # a.append(a[-1])
#             # time.append(time[-1]+0.38)
#             print('Wrong dimensions, estimated time:', time[-1])
#         else:
#             amplitudes.append(DFT.ampls[low_cut:])
#             time.append(DFT.time/60)
#             print(DFT.time/60)
#     except:
#         print('Wrong dimensions. something went wrong')

# # Meshgrid for surface plots
# freqs = DFT_list[0].freqs[low_cut:]
# x,y = np.meshgrid(time, freqs)

# # convert to transposed array for plotting
# z = np.array(amplitudes).transpose()

# # # Create figure
# # fig = pyplot.figure()
# # ax = fig.gca(projection='3d')

# # # Plot surface and rotate for top view (with t=0 on left)
# # ax.plot_surface(x,y,z, cmap=cm.coolwarm)
# # ax.view_init(90, 270)

# pyplot.figure()
# pyplot.pcolormesh(x,y,z,cmap=cm.coolwarm)
# pyplot.title('Spectrogram for the LifeLine DFT data')
# pyplot.xlabel('Time [min]')
# pyplot.ylabel('')
# # pyplot.show()


# # Do some analysis
# max_amps = []
# max_freqs = []
# for amplitude in amplitudes:
#     max_amps.append(max(amplitude))
#     ind = amplitude.index(max_amps[-1])
#     max_freqs.append(freqs[ind])

# # Filter the data with a savitzky-golay filter
# f_max_amps = savgol_filter(max_amps, 51, 2)
# f_max_freqs = savgol_filter(max_freqs, 51, 2)


# # Plot max amps and freqs over time
# fig, ax1 = pyplot.subplots()
# pyplot.grid()
# ax1.plot(time, f_max_amps, 'C0')
# ax1.set_xlabel('Time [s]')
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('Amplitude', color='C0')
# ax1.tick_params('y', colors='C0')

# ax2 = ax1.twinx()
# ax2.plot(time, f_max_freqs, 'C1')
# ax2.set_ylabel('Frequency [Hz]', color='C1')
# ax2.tick_params('y', colors='C1')

# fig.tight_layout()
# pyplot.show()