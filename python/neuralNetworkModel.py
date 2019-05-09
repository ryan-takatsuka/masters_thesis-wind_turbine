import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class neuralNetworkModel:
	'''
	This class holds the neural network model for the turbine data.  The neural network
	model is custom developed from scratch to clearly show the steps required for
	fitting a model, and also allows for better customization to optimize the 
	detection algorithm.  This model uses ALL of the frequency data as inputs to 
	calculate the rotor state (good or bad).

	This model is designed to work with 3 neural network layers, but it can have any
	number of units in the first and second layer.  The 3rd layer is only one unit
	(because it is a single-class classification problem).  The first layer will most
	likely contain the frequency spectrum data of the acceleration.

	'''

	def __init__ (self, X, Y, hidden_layer_size, lambda0=1):
		'''
		Initialize the model. The input layer size and output layer size are assumed.
		The input and output data gets randomized and split into test/training data
		here.  lambda0 is the regularization parameter that is used to prevent
		overfitting (a higher value is less likely to produce an overfitting model).
		
		Args:
			X: An array (or multi-dimensional list) containing the amplitude
				data for each frame.  This is the experimental input data
			Y: An array (or list) containing the status of the turbine
				for the amplitude data with the corrseponding index
			hidden_layer_size: The amount of units in the hidden layer
			lambda0: [1] The regularization parameter
		'''

		# Normalize the data
		X = (X-np.mean(X, axis=0)) / np.std(X, axis=0)

		# Randomize the data
		self.X, self.Y = self.shuffle_data(np.array(X), np.array(Y))

		# Split the data into training and test data
		test_size = 0.2
		train_index = int(test_size * self.Y.size)

		# Split up into train and test data
		self.X_test = self.X[0:train_index,:]
		self.Y_test = self.Y[0:train_index]
		self.X_train = self.X[train_index:,:]
		self.Y_train = self.Y[train_index:]

		# Setup the NN parameters
		self.input_layer_size = self.X_train.shape[1] # 
		self.hidden_layer_size = hidden_layer_size # 25 hidden units
		self.num_labels = 1 # 10 labels, from 1 to 10
		self.lambda0 = lambda0


	def shuffle_data(self, X, Y):
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


	def train_model(self, maxiter=100, gtol=1e-8):
		'''
		Train the neural network model.
		
		Args:
			maxiters: [100] The maximum iterations during the optimization process
			gtol: [1e-8] The minimum change in gradient between iterations

		Returns:
			The final cost value
		'''

		# Initialize some random weight values.  These must be randomly
		# initialized and cannot be set to zero.
		initial_Theta1 = self.randInitializeWeights(self.input_layer_size, 
			self.hidden_layer_size)
		initial_Theta2 = self.randInitializeWeights(self.hidden_layer_size, 
			self.num_labels)
		initial_nn_params = np.append(initial_Theta1.flatten(), 
			initial_Theta2.flatten())

		# print(initial_Theta2)

		# Verify that the gradient calculated with the cost function is correct
		# gradientChecking(lambda0)

		# Group the arguments for the cost function
		costFun_args = (self.input_layer_size, self.hidden_layer_size, self.num_labels,
			self.X_train, self.Y_train, self.lambda0)

		# Define a callback function used to save some intermediate values
		# during the optimization process
		J_iters = [] # initialize a list to hold the iteration cost values
		iter_num = [0] # initialize a list to hold the iteration numbers
		def callbackFun(x): # Create a callback function to save inter results
			iter_num.append(iter_num[-1]+1) # Add to the iteration number list
			J = self.nnCostFunction(x, *costFun_args) # Calculate cost function value
			print(iter_num[-1],',', J) # print intermediate results
			J_iters.append(J) # add intermediate cost value to a list

		# Optimize using a nonlinear conjugate gradient algorithm
		print('Optimization process started\nIteration number , Cost function value')
		result = scipy.optimize.fmin_cg(self.nnCostFunction, initial_nn_params,
			fprime=self.nnCostFunctionGradient, args=costFun_args, maxiter=maxiter,
			retall=True, disp=True, full_output=True, callback=callbackFun,
			gtol=gtol)
		self.nn_params = result[0]
		self.J_iters = J_iters

		# Calculate internal network parameters
		self.nn_result = self.calculate_all_layers(self.X, self.Y)

		# Return the final cost value
		return self.J_iters[-1]


	def calculate_all_layers(self, X, Y):
		'''
		Calculate all of the layer outputs.  This function returns all of the 
		intermediate variables used during the output calculation of the neural network.
		
		Args:
			X: The input data of size [m,n] where m is the number of training
				examples and n is the number of parameters
			Y: The output data [m,1] where m is the number of training examples

		Returns:
			A structure containing the intermediate variables from the neural
			network calculation.
		'''

		# Reshape the weight matrices into the proper dimensions
		Theta1 = np.reshape(self.nn_params[0:self.hidden_layer_size*(self.input_layer_size+1)], 
			(self.hidden_layer_size, self.input_layer_size+1))
		Theta2 = np.reshape(self.nn_params[self.hidden_layer_size*(self.input_layer_size+1):], 
			(self.num_labels, self.hidden_layer_size+1))

		# Calculate the number of training examples
		m = X.shape[0]

		# Calculate the hypothesis and intermediate activation layers
		a1 = np.column_stack([np.ones((m,1)), X]) # Add the bias unit to the data
		z2 = np.matmul(a1, Theta1.T) # Calculate z in layer 2
		a2 = self.sigmoid(z2) # Calculate the activations in layer 2
		a2 = np.column_stack([np.ones((m,1)), a2]) # Add the bias unit to the data
		z3 = np.matmul(a2, Theta2.T) # Calculate the z in layer 3 (the output layer)
		a3 = self.sigmoid(z3) # Calculate the activations in the third layer
		h = a3 # The hypothesis is the third and final layer

		# Calculate the cost
		J = self.nnCostFunction(self.nn_params, self.input_layer_size, 
			self.hidden_layer_size, self.num_labels,
			X, Y, self.lambda0)

		# Calculate the parameters
		result = {
			"Theta1": Theta1,
			"Theta2": Theta2,
			"m": m,
			"a1": a1,
			"a2": a2,
			"a3": a3,
			"z2": z2,
			"z3": z3,
			"h": h,
			"J": J
		}

		return result


	def sigmoidGradient(self, x):
		''' 
		Calculate the sigmoid gradient. This is the derivative of the
		sigmoid function.

		Args:
			x: The input variable or vector or matrix

		Returns:
			The gradient of the sigmoid function at x
		'''

		return self.sigmoid(x) * (1-self.sigmoid(x))


	def sigmoid(self, x):
		'''
		Calculate the sigmoid function.

		Args:
			x: The input variable or vector or matrix

		Returns:
			The value of the sigmoid function.
		'''

		return 1 / (1 + np.exp(-x))


	def nnCostFunction(self, nn_params, input_layer_size, hidden_layer_size, num_labels,
		X, y, lambda0):
		''' 
		The cost function of the neural network.  This only outputs the value of the cost
		function and not the gradient.

		Args:
			nn_params: the Theta1 and Theta2 variables for a 3-layer neural network flattened
				into a single vector
			input_layer_size: The number of features in the input layer (not including the bias unit)
			hidden_layer_size: The number of units in the hidden layer (not including the bias unit)
			num_labels: The number of classes (code needs to be modified to accomodate more than 1 class)
			X: The input data of size [m,n] where m is the number of training examples and n=input_layer_size
			y: The output data of size [m,1]
			lambda0: the regularization parameter

		Returns:
			The value of the cost function
		'''

		# Reshape the weight matrices into the proper dimensions
		Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
		Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))

		# Calculate the number of training examples
		m = X.shape[0]

		# Calculate the hypothesis and intermediate activation layers
		a1 = np.column_stack([np.ones((m,1)), X]) # Add the bias unit to the data
		z2 = np.matmul(a1, Theta1.T) # Calculate z in layer 2
		a2 = self.sigmoid(z2) # Calculate the activations in layer 2
		a2 = np.column_stack([np.ones((m,1)), a2]) # Add the bias unit to the data
		z3 = np.matmul(a2, Theta2.T) # Calculate the z in layer 3 (the output layer)
		a3 = self.sigmoid(z3) # Calculate the activations in the third layer
		h = a3 # The hypothesis is the third and final layer

		# Calculate the matrix split into classes
		y_matrix = np.zeros((m, num_labels))

		for idx in range(m):
			# y_matrix[idx,np.mod(y[idx], num_labels)] = 1
			y_matrix[idx] = y[idx]

		# Calculate the cost using matrix operations
		J = np.sum(-y_matrix * np.log(h) - (1-y_matrix) * np.log(1-h)) / m

		# Calculate the regularization term for the cost function
		J_reg = lambda0/2/m * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

		# Calculate the regularized cost
		J = J + J_reg

		return J


	def nnCostFunctionGradient(self, nn_params, input_layer_size, hidden_layer_size, 
		num_labels, X, y, lambda0):
		'''
		Calculate the gradient of the cost function analytically.  This is much faster than
		using the finite difference method to numerically calculate the gradient.

		Args:
			nn_params: the Theta1 and Theta2 variables for a 3-layer neural network flattened
				into a single vector
			input_layer_size: The number of features in the input layer (not including the bias unit)
			hidden_layer_size: The number of units in the hidden layer (not including the bias unit)
			num_labels: The number of classes (code needs to be modified to accomodate more than 1 class)
			X: The input data of size [m,n] where m is the number of training examples and n=input_layer_size
			y: The output data of size [m,1]
			lambda0: the regularization parameter

		Returns:
			The value of the cost function gradient, which is a flattened matrix equal to the size of nn_params

		'''

		# Reshape the weight matrices into the proper dimensions
		Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],
			(hidden_layer_size, input_layer_size+1))
		Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
			(num_labels, hidden_layer_size+1))

		# Calculate the number of training examples
		m = X.shape[0]

		# Calculate the hypothesis and intermediate activation layers
		a1 = np.column_stack([np.ones((m,1)), X]) # Add the bias unit to the data
		z2 = np.matmul(a1, Theta1.T) # Calculate z in layer 2
		a2 = self.sigmoid(z2) # Calculate the activations in layer 2
		a2 = np.column_stack([np.ones((m,1)), a2]) # Add the bias unit to the data
		z3 = np.matmul(a2, Theta2.T) # Calculate the z in layer 3 (the output layer)
		a3 = self.sigmoid(z3) # Calculate the activations in the third layer
		h = a3 # The hypothesis is the third and final layer

		# Calculate the matrix split into classes
		y_matrix = np.zeros((m, num_labels))

		for idx in range(m):
			# y_matrix[idx,np.mod(y[idx], num_labels)] = 1
			y_matrix[idx] = y[idx]

		# Calculate delta
		delta3 = a3 - y_matrix
		delta2 = np.matmul(delta3, Theta2[:,1:]) * self.sigmoidGradient(z2)

		# Calculate Delta
		Delta1 = np.matmul(delta2.T, a1)
		Delta2 = np.matmul(delta3.T, a2)

		# Calculate the gradient
		Theta1_grad = Delta1 / m
		Theta2_grad = Delta2 / m

		# Initialize the regularization term.  
		# This initializes the first element that is set to zero
		Theta1_grad_reg = np.zeros(Theta1_grad.shape)
		Theta2_grad_reg = np.zeros(Theta2_grad.shape)

		# Calculate the regularization without calculating for the bias unit
		Theta1_grad_reg[:,1:] = lambda0 / m * Theta1[:,1:]
		Theta2_grad_reg[:,1:] = lambda0 / m * Theta2[:,1:]

		# Add the regularization term to the gradient
		Theta1_grad = Theta1_grad + Theta1_grad_reg
		Theta2_grad = Theta2_grad + Theta2_grad_reg

		# Set the unrolled gradient output
		grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())

		return grad		


	def randInitializeWeights(self, N_in, N_out):
		''' Perform gradient checking 
		
		Args:
			N_in (scalar): The number of input units for the NN
			N_out (scalar): THe number of output units for the NN. 
				This is also equivalent to the number of classes in 
				a classification problem

		Returns:
			array: A randomly initialized array with size [N_out, N_in+1]

		'''

		# Estimate alpha using the input and output layer sizes
		# This expression is an effective strategy for choosing epsilon
		epsilon_init = np.sqrt(6) / (np.sqrt(N_out + N_in+1))
		# epsilon_init = 0.12

		# Create the random array of initialization weights
		W = np.random.rand(N_out, N_in+1) * 2 * epsilon_init - epsilon_init

		return W


	def gradientChecking(self, lambda0=0):
		'''
		Perform gradient checking.  This is a way of verifying the analytic gradient
		is accurate and implemented correctly.  This method should not be used after
		the gradient is verified to be correct.

		Args:
			lambda0: [0] The regularization parameter

		Returns:
			A flag determining if the gradient is calculated correctly
		'''

		# Define a small neural network to perform the gradient checking on
		input_layer_size = 3 # number of units in the input layer
		hidden_layer_size = 5 # number of units in the hidden layer
		num_labels = 3 # number of classes
		m = 5 # number of training examples

		# Initialize some semi-random test data.  Initializing using "sin" 
		# will ensure that the weights are always of the same values and will
		# be useful for debugging.  Completely "random" values will change 
		# every time the function is run, making it hard to track down errors
		Theta1 = np.zeros((hidden_layer_size, input_layer_size+1)) # initialize
		Theta2 = np.zeros((num_labels, hidden_layer_size+1)) # initialize
		Theta1 = np.reshape(np.sin(range(Theta1.size)), Theta1.shape)
		Theta2 = np.reshape(np.sin(range(Theta2.size)), Theta2.shape)

		# Generate a semi-random X and y matrices
		X = np.zeros((m, input_layer_size)) # initialize
		y = np.zeros(m) # initialize
		X = np.reshape(np.sin(range(X.size)), X.shape)
		y = 1 + np.mod(range(m), num_labels)

		# unroll the parameters
		nn_params = np.append(Theta1.flatten(), Theta2.flatten())

		# Group the arguments used by the cost function.  A pointer to these
		# variables will be used to simplify the notation.
		costFun_args = (input_layer_size, hidden_layer_size, num_labels,
			X, y, lambda0)

		# Check the 2 gradient calculation methods
		diff = scipy.optimize.check_grad(self.nnCostFunction, 
			self.nnCostFunctionGradient, nn_params, *costFun_args, epsilon=1e-7)

		# Calculate the normalized difference between the 2 solutions.  This 
		# evaluates how close the calculations of the 2 methods are.  If this
		# value is less than 1e-5, then the analytic method has been successfully
		# verified.
		successful_diff = 1e-5 # difference must be smaller than this
		if diff < successful_diff:
			gradient_good = True
			print('The cost function gradient calculation is functional!')
		else:
			gradient_good = False
			print('There is a problem with the cost function gradient calculation')

		print('Difference: ', diff)
		return gradient_good


	def predict(self, X=None, Y=None):
		'''
		Predict new outputs using the trained model

		Args:
			X: The input data (uses test data by default)
			Y: The output data (uses test data by default)

		Returns:
			The predictions and the accuracy score
		'''

		# Set default X and Y
		if X is None or Y is None:
			X = self.X_test
			Y = self.Y_test

		nn = self.calculate_all_layers(X, Y)

		# Calculate the result for single/multi-class outputs
		# result = np.argmax(nn["h"], axis=1)+1 # multi-class
		predictions = np.round(nn["h"]) # single-class

		# Calculate the accuracy of the prediction
		accuracy = np.mean(predictions.flatten()==Y.flatten()) * 100

		return predictions, accuracy


	def plot_iteration(self):
		'''
		Visualize and plot the cost function during the iteration process.
		By default, no plots are shown until plt.show() is called.
		'''

		plt.figure()
		plt.plot(list(range(len(self.J_iters))), self.J_iters)
		plt.title('Cost function during optimization')
		plt.xlabel('Number of iterations')
		plt.ylabel('Cost function value')


	def plot_decision_boundary(self, X, Y, num_points=200):
		'''
		Plot the decision boundary.  This only works if there are 2 units
		in the hidden layer.
		'''
		
		# There must only be 2 units in the hidden layer
		assert self.hidden_layer_size == 2

		# Calculate the internal results		
		nn_result = self.calculate_all_layers(X, Y)

		# Create color maps used for plotting classifications
		cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
		cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


		a2 = nn_result["a2"][:,1:3]
		a2 = nn_result["z2"]


		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].  Set up these range
		# values so that they cover a slightly wider range than the data
		range_increase_perc = 0.2 # amount to expand the limits of the data
		x_min, x_max = a2[:, 0].min(), a2[:, 0].max() # find limits of data
		y_min, y_max = a2[:, 1].min(), a2[:, 1].max() # find limits of data
		x_min = x_min - (x_max - x_min) * range_increase_perc # expand range
		x_max = x_max + (x_max - x_min) * range_increase_perc # expand range
		y_min = y_min - (x_max - x_min) * range_increase_perc # expand range
		y_max = y_max + (x_max - x_min) * range_increase_perc # expand range

		# Create the mesh grid used to create the decision boundary plot
		n = num_points
		xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
							 np.linspace(y_min, y_max, n))

		# Calculate the last layer assuming a known layer 2 activation for each unit.
		# These values are the decision boundary values for the plot
		a2_db = np.vstack((xx.ravel(), yy.ravel())).T
		a2_db = np.column_stack([np.ones((a2_db.shape[0],1)), a2_db]) # Add the bias unit to the data
		z3_db = np.matmul(a2_db, nn_result["Theta2"].T) # Calculate the z in layer 3 (the output layer)
		a3_db = self.sigmoid(z3_db) # Calculate the activations in the third layer
		h_db = np.round(a3_db)


		# Put the decision boundary in a color plot
		Z = h_db.reshape(xx.shape)
		plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		# Plot all of the experimental data
		plt.scatter(a2[:, 0], a2[:, 1], c=Y, cmap=cmap_bold,
					edgecolor='k', s=20)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("Neural Network Classification")
		plt.xlabel('Hidden layer unit #1 activation')
		plt.ylabel('Hidden layer unit #2 activation')
		plt.legend(('Good', 'Bad'))

		balanced = mpatches.Patch(color='#00FF00', label='Class A')
		not_balanced = mpatches.Patch(color='#FF0000', label='Class B')
		plt.legend(handles=[balanced, not_balanced])