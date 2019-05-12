'''
This file runs the custom logistic regression model for the 
experimental data.

Author: Ryan Takatsuka

'''


# Import stuff
import split_dft
import LifeLine_file
import simulate_LL_data
import logisticRegressionModel
import matplotlib.pyplot as plt
import numpy as np
import pandas


# The experimental data filename
filename = 'experimental_data//WDFT_2018-04-01_160352.txt'

# Read experimental data for an imbalanced rotor in a LifeLine file
exp_data = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=True, idx_range=[18, 73])
# Add experimental data to the training/testing dataset
dataset = []
exp_data.addto_dataset(dataset)

# Read experimental data for balanced rotor in a LifeLine file
exp_data2 = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=False, idx_range=[185, 240])
# Add simulated data to the training/testing dataset
exp_data2.addto_dataset(dataset)

# The names of the dataset variables
names = ['Frequency', 'Amplitude', 'status']

# Create a pandas dataset
dataset = pandas.DataFrame(dataset, columns=names)

# Set the X and Y parameters for the dataset
X = dataset.values[:,0:2]
Y = dataset.values[:,2]

# Add all frequency data to experimental data
X = []
Y = []
exp_data.add_all_variables(X, Y)
exp_data2.add_all_variables(X, Y)

# Create the detection model for determining an imbalance in the rotor
LR_model = logisticRegressionModel.logisticRegressionModel(X, Y, lambda0=1, order=1)

# Train the logistic regression model
result = LR_model.train_model()

# Make predictions with the test data
pred, accuracy = LR_model.predict(LR_model.X_test, LR_model.Y_test)
print('The model accuracy: ', accuracy, '%')

# Plot some results
LR_model.plot_iteration() # plot the cost function during the optimization process
# LR_model.plot_decision_boundary(LR_model.X, LR_model.Y, num_points=200, smooth=True) # db
LR_model.plot_theta(exp_data.frequencies)
plt.show()

