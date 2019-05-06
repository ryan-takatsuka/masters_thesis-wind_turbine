# Load libraries
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import numpy as np



class detectionModel:
	'''
	This class holds the model used to detect a rotor imbalance.  This model uses the maximum
	amplitude and corresponding frequency and determines whether the turbine status
	is good or bad (balanced or imbalanced).  The model is trained from knwon experimental
	data, and should become better with more data.
	'''

	def __init__ (self, dataset):
		'''
		Initialize the model.

		Args:
			dataset (list): The experimental training/testing dataset that is used to build
				the model
		'''

		# The names of the dataset variables
		names = ['Frequency', 'Amplitude', 'status']

		# Create a pandas dataset
		dataset = pandas.DataFrame(dataset, columns=names)

		# Set the X and Y parameters for the dataset
		X = dataset.values[:,0:2]
		Y = dataset.values[:,2]

		# The ratio of test data to use (The remaining is used for model training)
		test_size = 0.20

		# Random seed
		seed = 7

		# Split the data up into test and training sets
		self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

		# Create a label encoder used to transform the class data into numeric data
		self.le = preprocessing.LabelEncoder()

		# Intialize model
		self.model = KNeighborsClassifier(n_neighbors=3, weights='uniform')


	def test_different_models(self, plot=False):
		'''
		Test different classification algorithms to determine what seems to work the best.
		'''

		# Set seed and specify the scoring
		seed = 7
		scoring = 'accuracy'

		# Algorithms list
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNC', KNeighborsClassifier()))
		models.append(('DTC', DecisionTreeClassifier()))
		models.append(('GNB', GaussianNB()))
		models.append(('SVC', SVC()))

		# evaluate each model
		results = []
		names = []
		print('-------------')
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			print(msg)

		if plot:
			# Compare Algorithms
			fig = plt.figure()
			fig.suptitle('Algorithm Comparison')
			ax = fig.add_subplot(111)
			plt.boxplot(results)
			ax.set_xticklabels(names)


	def train(self):
		'''
		Train the model using data.
		'''

		# fit the KNeighborsClassifier model
		self.model.fit(self.X_train, self.Y_train)

		return self.model

	def test(self):
		'''
		Test the model by using the validation (test) part of the dataset.
		'''

		# Predict the turbine status
		predictions = self.model.predict(self.X_test)

		# Get validation/test statistics
		accuracy = accuracy_score(self.Y_test, predictions)
		confusion = confusion_matrix(self.Y_test, predictions)
		classification = classification_report(self.Y_test, predictions)

		print('--------------')
		print('Accuracy: %0.2f percent' % (accuracy*100))
		print('Confusion Matrix: \n', confusion)
		print('Classification Report: \n', classification)

		return predictions


	def plot_model(self):
		'''
		Plot the classification borders.  This is only useful with 2 (maybe 3) dimensional data because it is pretty hard to visualize higher dimensions.  The trained model
		is plotted as color areas on a chart, and the individual samples are plotted as
		points.
		'''

		# Create color maps used for plotting classifications
		cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
		cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

		# Mesh step size
		h = 0.02

		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
		y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
		Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = self.le.fit_transform(Z).reshape(xx.shape)
		plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


		# Plot also the training points
		plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.le.fit_transform(self.Y_train), cmap=cmap_bold,
					edgecolor='k', s=20)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("2-Class classification (k = %i, weights = '%s')"
				  % (self.model.n_neighbors, self.model.weights))
		plt.xlabel('Maximum Peak Frequency [Hz]')
		plt.ylabel('Maximum Peak Amplitude')
		plt.legend(('Class A', 'Class B'))

		balanced = mpatches.Patch(color='#00FF00', label='Class A')
		not_balanced = mpatches.Patch(color='#FF0000', label='Class B')
		plt.legend(handles=[balanced, not_balanced])