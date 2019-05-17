'''
This file runs the sklearn machine learning algorithms on the
experimental data.  This is the KNN algorithm used in the Thesis Report.

Author: Ryan Takatsuka

'''

# Import stuff
import split_dft
import LifeLine_file
import simulate_LL_data
import detectionModel
import matplotlib.pyplot as plt
import numpy as np

# The experimental data filename
filename = 'experimental_data//WDFT_2018-04-01_160352.txt'

# Read experimental data for an imbalanced rotor in a LifeLine file
exp_data = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=True, idx_range=[18, 73])
# exp_data.plot() # Plot the data

# Add experimental data to the training/testing dataset
dataset = []
exp_data.addto_dataset(dataset)

# Read experimental data for balanced rotor in a LifeLine file
exp_data2 = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=False, idx_range=[185, 240])
# exp_data2.plot() # plot the data

# Add simulated data to the training/testing dataset
exp_data2.addto_dataset(dataset)

# Create the detection model for determining an imbalance in the rotor
my_algorithm = detectionModel.detectionModel(dataset)

# Train and validate the model
model = my_algorithm.train()
predictions = my_algorithm.test()

# Plot model
my_algorithm.plot_model()

# Read experimental data for balanced rotor in a LifeLine file
all_data = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=False)
all_data.plot() # plot the data



# Show the plots
plt.show()
