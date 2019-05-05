import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class simulated_data:
	'''
	I don't have any data for a rotor imbalance, so I'll just simulate it for now in order to
	train the classification algorithm.
	I insert noise for some randomness, but the data is probably much more messy in real life. 
	Hopefully this simulated data is at least somewhat similar to actual imbalance data.
	'''

	def __init__ (self, time=[]):
		'''
		Initialze the simulate data.

		Args:
			time (list): Include a time vector so the simulated data is the same length as 
				actual data.	This is not required, and will default to a 300-element list from
				0-120 minutes.
		'''

		# Create the time vector if there was no input
		if not(time):
			# 300-element list from 0-120 minutes
			self.time = np.linspace(0,120,300)
		else:
			self.time = time

		# Initialize some parameters
		self.max_freqs = []
		self.max_amps = []
		self.is_balanced = []

	def simulate_imbalance(self):
		'''
		Simulate a rotor imbalance by assuming that the rotor frequency is oscillating about
		a certain value, while the amplitude is constant.  All of the simulated data has noise
		added.
		'''

		# Create max amplitude and frequency vectors for a simulated imbalance
		self.max_freqs = 0.5 * np.sin(.02*2*np.pi*np.array(self.time)) + 2 + 0.2*np.random.randn(len(self.time))

		for f in self.max_freqs:
			self.max_amps.append(np.asscalar(0.6*f + 0.5*np.random.randn(1)))
		# self.max_amps = 1 * np.ones(len(self.time)) + 0.1*np.random.randn(len(self.time))

		# Specify as imbalanced data
		self.is_balanced = False


	def plot(self):
		fig, ax1 = plt.subplots()
		plt.grid()
		ax1.plot(self.time, self.max_amps, 'C0')
		ax1.set_xlabel('Time [min]')
		# Make the y-axis label, ticks and tick labels match the line color.
		ax1.set_ylabel('Amplitude', color='C0')
		ax1.set_title('Simulated Rotor Imbalance')
		ax1.tick_params('y', colors='C0')

		ax2 = ax1.twinx()
		ax2.plot(self.time, self.max_freqs, 'C1')
		ax2.set_ylabel('Frequency [Hz]', color='C1')
		ax2.tick_params('y', colors='C1')
		fig.tight_layout()


	def addto_dataset(self, dataset):
		'''
		Add to the training/testing dataset for the classification algorithm.

		Args:
			dataset (list): The dataset to add to

		Returns:
			The appended dataset.
		'''

		if self.is_balanced:
			category = 'good'
		else:
			category = 'bad'

		for ind,t in enumerate(self.time):
			dataset.append([self.max_freqs[ind], self.max_amps[ind], category])

		return dataset

	def filter(self, ftype='SG', order=2, framelen=51, wn=0.1):
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
			self.remove_transient_samples(20)


		if ftype=='Window_FIR':
			# Design the filter (firwin uses number of taps, which is order+1)
			b = signal.firwin(order+1, wn)

			# Apply the FIR filter
			self.max_amps = signal.lfilter(b, 1, self.max_amps)
			self.max_freqs = signal.lfilter(b, 1, self.max_freqs)	

			# Remove the transient samples
			self.remove_transient_samples(10)

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

