# Import stuff
import split_dft
import LifeLine_file
import simulate_LL_data
import detectionModel
import matplotlib.pyplot as plt
import numpy as np
import neuralNetworkModel


# Filenames with experimental data
filename = 'experimental_data//WDFT_2018-04-01_160352.txt'

# Neural network variable initialization
X = []
Y = []

# Read experimental data for an imbalanced rotor in a LifeLine file
exp_data = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=True, idx_range=[18, 73])

# Add experimental data to the training/testing dataset
exp_data.add_all_variables(X, Y)

# Read experimental data for balanced rotor in a LifeLine file
exp_data2 = LifeLine_file.LifeLine_file(filename, low_cut=42,
	is_balanced=False, idx_range=[185, 240])

# Add simulated data to the training/testing dataset
exp_data2.add_all_variables(X, Y)

# Create the neural network model
hidden_layer_size = 2
nnModel = neuralNetworkModel.neuralNetworkModel(X, Y, 
	hidden_layer_size, lambda0=1)

# Train the neural network model
nnModel.train_model()

# Calculate the model accuracy for the test data
predictions, accuracy = nnModel.predict(X=nnModel.X_test, Y=nnModel.Y_test)
print('The model accuracy: ', accuracy, '%')

# Plot some results
nnModel.plot_iteration() # plot the cost function during the optimization process
nnModel.plot_decision_boundary(nnModel.X_test, nnModel.Y_test, num_points=200) # db
# plt.show()

print(nnModel.nn_result['Theta1'][0])

plt.figure()
plt.plot(exp_data.frequencies, nnModel.nn_result['Theta1'][0,1:])
plt.plot(exp_data.frequencies, nnModel.nn_result['Theta1'][1,1:])
plt.show()