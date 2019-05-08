from __future__ import absolute_import, division, print_function
import LifeLine_file

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(X, Y):
	''' 
	Shuffle the data to randomize it by rows.  The experimental data
	needs to be randomized before it can be split up and processed.

	Args:
		X: The array of input data
		Y: The output data

	Returns:
		tuple: the new X and Y arrays that are randomly shuffled.
	'''

	# Verify both inputs have the same number of rows
	assert X.shape[0] == Y.shape[0]

	# Set the number of rows
	num_rows = X.shape[0]

	# Create random indexing
	p = np.random.permutation(num_rows)

	# Return new variables with randomized rows synchronized between
	# the variables
	return X[p], Y[p]


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

# Normalize the data
X = (X-np.mean(X, axis=0)) / np.std(X, axis=0)

# Randomize the data
X, Y = shuffle_data(np.array(X), np.array(Y))

# Split the data into training and test data
test_size = 0.2
train_index = int(test_size * Y.size)

# Split up into train and test data
X_test = X[0:train_index,:]
Y_test = Y[0:train_index]
X_train = X[train_index:,:]
Y_train = Y[train_index:]

model = keras.Sequential([
	# keras.layers.Dense(1, input_shape=(X.shape[1],)),
	keras.layers.Dense(100, activation='sigmoid', input_shape=(X.shape[1],)),
	keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', 
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(X_test)

print(model.summary())
print(model.layers[0].get_weights()[1])

plt.figure()
for i in range(2):
	plt.plot(model.layers[0].get_weights()[0][:,i])
plt.show()