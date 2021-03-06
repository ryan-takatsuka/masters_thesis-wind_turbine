# Load libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import scipy.optimize
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures
import my_colors



class logisticRegressionModel:
	'''
	This class holds the Logistic Regression model used to detect a rotor imbalance.  This
	algorithm is built from scratch, and uses matrix math to train and validate the model.
	Building the model from scratch allows for many possible customizations.
	'''

	def __init__ (self, X, Y, lambda0=1, order=1):
		'''
		Initialize the model.

		Args:
			dataset (list): The experimental training/testing dataset that is used to build
				the model
			lambda0: [1] The regularization parameter
			order: The polynomial order to fit
		'''

		# Set the regularization parameter
		self.lambda0 = lambda0

		# Define the polynomial used to fit the data and modify the 
		# input data to accomodate this model
		self.polynomial_features= PolynomialFeatures(degree=order)
		X = self.polynomial_features.fit_transform(X)[:,1:]

		# Normalize the data
		# X = (X-np.mean(X, axis=0)) / np.std(X, axis=0)

		# Change output to binary values
		Y = preprocessing.LabelEncoder().fit_transform(Y)

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


	def shuffle_data(self, X, Y):
		''' 
		Shuffle the data to randomize it by rows.

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
		return np.float64(X[p]), np.float64(Y[p])


	def costFunction(self, theta, X, Y, lambda0):
		''' The logistic regression cost function '''

		# Determine the number of training examples
		m = Y.size

		# Create a matrix from Y
		Y = np.matrix(Y)

		# Calculate the X matrix and concatenate the bias ones
		X = np.hstack([np.ones((m, 1)), np.array(X)])

		# Calculate the hypothesis
		h = self.sigmoid(np.matmul(X, theta))

		# Calculate the cost
		J = 1 / m * np.sum(np.multiply(-Y,np.log(h)) - np.multiply((1-Y),np.log(1-h))) + \
			lambda0/2/m * np.dot(theta[1:].T, theta[1:])

		return J


	def costFunctionGradient(self, theta, X, Y, lambda0):
		''' Calculate the gradient of the cost function '''

		# Determine the number of training examples
		m = Y.size

		# Calculate the X matrix and concatenate the bias ones
		X = np.hstack([np.ones((m, 1)), np.array(X)])

		# Calculate the hypothesis
		h = self.sigmoid(np.matmul(X, theta))

		# Create a theta variable with the first unit initialized to zero
		theta_0 = np.append(0, theta[1:])

		# Calculate the gradient of the cost function
		grad = 1/m * np.matmul((h-Y).T, X) + lambda0/m*theta_0

		return grad


	def sigmoidGradient(self, x):
		''' Calculate the sigmoid Gradient '''

		return self.sigmoid(x) * (1-self.sigmoid(x))


	def sigmoid(self, x):
		''' Calculate the sigmoid function '''

		return 1 / (1 + np.exp(-x))


	def train_model(self, maxiter=100, tol=1e-8):
		''' Train the logistic regression model '''

		# Initialize the parameters to 0
		theta_init = np.zeros((self.X_train.shape[1]+1,1))

		# Group the arguments for the model to be trained
		costFun_args = (self.X_train, self.Y_train, self.lambda0)

		# Define a callback function used to save some intermediate values
		# during the optimization process
		self.J_iters = [] # initialize a list to hold the iteration cost values
		iter_num = [0] # initialize a list to hold the iteration numbers
		def callbackFun(x): # Create a callback function to save inter results
			iter_num.append(iter_num[-1]+1) # Add to the iteration number list
			J = self.costFunction(x, *costFun_args) # Calculate cost function value
			print(iter_num[-1],',', J) # print intermediate results
			self.J_iters.append(J) # add intermediate cost value to a list

		# Specify options for the minimization function
		options = {"maxiter": maxiter}

		# Optimize the parameters
		result = scipy.optimize.minimize(self.costFunction, theta_init, 
			args=costFun_args, jac=self.costFunctionGradient, 
			callback=callbackFun, options=options, tol=tol)

		# Set the output parameters
		self.theta = np.matrix(result.x).T

		return result


	def predict(self, X, Y):
		''' Predict new values and calculate accuracy '''

		# Determine the number of training examples
		m = Y.size

		# Calculate the X matrix and concatenate the bias ones
		X = np.hstack([np.ones((m, 1)), np.array(X)])

		# Calculate the hypothesis
		h = self.sigmoid(np.matmul(X, self.theta))

		# Determine predictions
		predictions = np.round(h)

		# Calculate the accuracy of the prediction
		accuracy = np.mean(predictions.flatten()==Y.flatten()) * 100

		return predictions, accuracy


	def calculate_hypothesis(self, X):
		''' Calculate the hypothesis with a trained model '''

		# Determine the number of training examples
		m = X.shape[0]

		# Calculate the X matrix and concatenate the bias ones
		X = np.hstack([np.ones((m, 1)), np.array(X)])

		# Calculate the hypothesis
		h = self.sigmoid(np.matmul(X, self.theta))

		return h


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


	def plot_decision_boundary(self, X, Y, num_points=1000, smooth=False):
		'''
		Plot the decision boundary.  This only works if there are 2 units
		in the hidden layer.
		'''

		# Create color maps used for plotting classifications
		discrete_colors = ['#FFAAAA', '#AAFFAA']
		smooth_colors = my_colors.linear_gradient('#FFAAAA', finish_hex='#AAFFAA', n=100)['hex']

		# Create the colormap from the list
		if smooth: # create a smooth colormap proportional to the probability
			cmap_light = ListedColormap(smooth_colors)
		else: # create a discrete colormap line
			cmap_light = ListedColormap(discrete_colors)

		# Map the data points with a darker color
		cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].  Set up these range
		# values so that they cover a slightly wider range than the data
		range_increase_perc = 0.2 # amount to expand the limits of the data
		x_min, x_max = X[:, 0].min(), X[:, 0].max() # find limits of data
		y_min, y_max = X[:, 1].min(), X[:, 1].max() # find limits of data
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
		X0 = np.vstack((xx.ravel(), yy.ravel())).T

		# Define the polynomial used to fit the data and modify the 
		# input data to accomodate this model (don't use the bias column
		# because we will make our own)
		X_order = self.polynomial_features.fit_transform(X0)[:,1:]

		# Calculate output of the model with the specified order
		# Z0 = np.array(np.round(self.calculate_hypothesis(X_order)))
		Z0 = np.array(self.calculate_hypothesis(X_order))

		# Put the decision boundary in a color plot
		Z0 = Z0.reshape(xx.shape)
		plt.figure()
		plt.pcolormesh(xx, yy, Z0, cmap=cmap_light)

		Y = [int(np_float) for np_float in Y]
		Y = np.array(Y)

		# Plot all of the experimental data
		plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
					edgecolor='k', s=20)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("Logistic Regression Decision Boundary")
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Acceleration at peak frequency component [m/s/s]')
		plt.legend(('Good', 'Bad'))

		balanced = mpatches.Patch(color='#00FF00', label='Class A')
		not_balanced = mpatches.Patch(color='#FF0000', label='Class B')
		plt.legend(handles=[balanced, not_balanced])


	def plot_theta(self, freqs):
		''' plot the parameters of the model, theta '''		

		plt.figure()
		plt.plot(freqs, self.theta[1:])
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Parameter Value')
		plt.title('Optimized Model Parameters')
