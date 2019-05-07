from matplotlib import pyplot as plt
from matplotlib import cm
import split_dft
from scipy import signal
import numpy as np
import pdb


class LifeLine_file:
	'''
	This class holds all the data for a single LifeLine file.  It includes
	a list of DFT_data objects for each DFT instance in the file.

	Attributes:
		filename (string): The filename for the LifeLine data
		DFT_list (list): A list of `DFT_data` objects
		max_amps (list): A list of the maximum amplitudes (one for each DFT)
		max_freqs (list): [Hz] A list of the corresponding frequencies for the maximum amplitudes
		amplitudes (list): A list containing lists for the good amplitudes in each of the DFTs
		frequencies (list): [Hz] A list of the frequencies used in each DFT
		time (list): [minutes] A list of the times for each DFT

	'''

	def __init__ (self, filename, is_balanced=True, low_cut=50, idx_range=[0,100]):
		'''
		Initialize the LifeLine_file object.

		Args:
			filename (string): The filename of the LifeLine file

		Notes:
			The directory name `dir_name` is not used because plotting is disabled in
			the `split_up` function.
		'''

		# Define the filename
		self.file_name = filename

		# Initialize some more attributes
		self.amplitudes = [] # A list of the amplitudes for the GOOD data
		self.frequencies = [] # A list of the frequencies
		self.time = [] # A list of the times for the GOOD data
		self.max_amps = [] # list of max amplitudes
		self.max_freqs = [] # list of corresponding frequency to max amplitude
		self.rotor_freq = [] # list of the average rotor frequency

		# Label this file as balanced or imbalanced turbine data
		self.is_balanced = is_balanced

		# The directory name is not used because we are not plotting the data
		dir_name = 'not_used'

		# Open the big file which is to be split up, then split it into DFT data sets.
		try:
			with open (self.file_name, mode = 'r') as big_file:
				print ('File opened.')
				self.DFT_list = split_dft.split_up (big_file, dir_name, plots = False)
				self.DFT_list = self.DFT_list[idx_range[0]:idx_range[1]]
				# pdb.set_trace()
				print ('Size of DFT list: {:d} items'.format (len(self.DFT_list)))

		except FileNotFoundError:
			print ('File "' + self.file_name + '" does not exist.')

		self.calc_max_values(low_cut)


	def calc_max_values(self, low_cut):
		'''
		Calculate the maximum amplitude and corresponding frequency for each DFT list.

		Args:
			low_cut (scalar): [default: 25] Cutoff the first few DFT points near the DC range.
				The DC component is usually the strongest component, so it needs to be removed. 

		Returns:
			A list containing the maximum amplitude and corresponding frequency for each DFT
		'''

		# Get the starting time
		start_time = self.DFT_list[0].time/60

		# Iterate through the DFTs and add the good data to some lists
		for DFT in self.DFT_list:
			try:
				if len(DFT.ampls)!=256:
					# There is not enough frequency points in this DFT
					pass
				else:
					self.amplitudes.append(DFT.ampls[low_cut:])
					self.time.append(DFT.time/60 - start_time) # convert to minutes
					self.rotor_freq.append(DFT.rotor_mean/60) # convert to Hz
			except:
				print('Bad DFT Data')

		# Frequency data should be the same for each DFT, so just use the first one
		self.frequencies = self.DFT_list[0].freqs[low_cut:]
		# print(self.frequencies[-1])
		# print(len(self.DFT_list[0].freqs))

		# Calculate the maximum amplitude and frequency
		for amplitude in self.amplitudes:
			self.max_amps.append(max(amplitude))
			ind = amplitude.index(self.max_amps[-1])
			self.max_freqs.append(self.frequencies[ind])


	def filter(self, ftype='SG', order=2, framelen=41, wn=0.1):
		'''
		This method filters the `max_amps` and `max_freqs` data using the specified filter

		Args:
			type ('string'): An optional parameter describing the type of filter to use.

				- Savitzky-Golay filter ('SG'), [order=2, framelen=51]
				- Butterworth IIR filter ('Butter_IIR') [order=2, wn=0.1]
				- Equiripple FIR filter ('Equiripple_FIR') [order=2, wn=0.1]

		Todo:
			Add the remaining filtering options
		'''

		if ftype=='SG':
			# Filter the data with a savitzky-golay filter
			self.max_amps = signal.savgol_filter(self.max_amps, framelen, order)
			self.max_freqs = signal.savgol_filter(self.max_freqs, framelen, order)

		if ftype=='Butter_IIR':
			# Design the butterworth filter
			b, a = signal.butter(order, wn)

			# Apply the butterworth filter
			self.max_amps = signal.lfilter(b, a, self.max_amps)
			self.max_freqs = signal.lfilter(b, a, self.max_freqs)

			# Remove the transient samples
			self.remove_transient_samples(10)

		if ftype=='Window_FIR':
			# Design the filter (firwin uses numtaps, which is order+1)
			b = signal.firwin(order+1, wn)

			# Apply the FIR filter
			self.max_amps = signal.lfilter(b, 1, self.max_amps)
			self.max_freqs = signal.lfilter(b, 1, self.max_freqs)	

			# Remove the transient samples
			self.remove_transient_samples(10)


	def plot(self):
		'''
		Visualize the data in plots.  This plots both a spectrogram and the max amplitude/
		frequency over time for the DFT data in the file.
		'''

		# Meshgrid for surface plots
		x,y = np.meshgrid(self.time, self.frequencies)

		# convert to transposed array for plotting
		z = np.array(self.amplitudes).transpose()

		# Create a spectrogram plot (essential a 2D surface plot)
		plt.figure()
		plt.pcolormesh(x,y,z,cmap=cm.coolwarm)
		plt.title('Spectrogram for the LifeLine DFT data')
		plt.xlabel('Time [min]')
		plt.ylabel('Frequency [Hz]')
		plt.colorbar()

		# Plot max amps and freqs over time
		fig, ax1 = plt.subplots()
		plt.grid()
		ax1.plot(self.time, self.max_amps, 'C0')
		ax1.set_xlabel('Time [min]')
		# Make the y-axis label, ticks and tick labels match the line color.
		ax1.set_ylabel('Amplitude', color='C0')
		ax1.tick_params('y', colors='C0')

		ax2 = ax1.twinx()
		ax2.plot(self.time, self.max_freqs, 'C1')
		# ax2.plot(self.time, self.rotor_freq, '--')
		ax2.set_ylabel('Frequency [Hz]', color='C1')
		ax2.tick_params('y', colors='C1')
		fig.tight_layout()


	def addto_dataset(self, dataset):
		'''
		Add the LifeLine file data to the machine learning dataset

		Args:
			dataset (list): A list used to test and train a classification algorithm

		Returns:
			The dataset with appended values from this LifeLine file
		'''

		if self.is_balanced:
			status = 'good'
		else:
			status = 'bad'

		for ind,t in enumerate(self.time):
			dataset.append([self.max_freqs[ind], self.max_amps[ind], status])

		# return dataset


	def remove_transient_samples(self, num_samples):
		'''
		Some of the real-time filters (like IIR and FIR) take a few samples to converge.
		This is because the filter states are initialized as 0, and must be calculated.
		This transient initialization phase creates misleading data for the classification
		algorithm and should be removed.

		Args:
			num_samples (scalar): The number of samples at the beginning of the dataset
				to remove. 
		'''

		self.max_amps = self.max_amps[num_samples:]
		self.max_freqs = self.max_freqs[num_samples:]
		self.time = self.time[num_samples:]
		self.amplitudes = self.amplitudes[num_samples:]


	def add_all_variables(self, X, Y):
		'''
		Create the neural network variables from the data.
		'''

		if self.is_balanced:
			status = 1 # good
		else:
			status = 0 # bad

		Xi = np.array(self.amplitudes)

		for ind,t in enumerate(self.time):
			X.append(self.amplitudes[ind])
			Y.append(status)




if __name__=='__main__':
	import simulate_LL_data

	myfile = LifeLine_file('full_data.txt')
	myfile.filter()
	myfile.plot()
	# plt.show()

	dataset = []
	myfile.addto_dataset(dataset)

	sim_data = simulate_LL_data.simulated_data(myfile.time)
	sim_data.simulate_imbalance()
	sim_data.plot()
	sim_data.addto_dataset(dataset)

