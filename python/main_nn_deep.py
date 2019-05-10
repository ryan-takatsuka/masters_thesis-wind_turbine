# Import stuff
import split_dft
import LifeLine_file
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.utils import shuffle

# Filenames with experimental data
filename = 'experimental_data//WDFT_2018-04-01_160352.txt'

# Neural network variable initialization
X = []
Y = []

# Read experimental data for an imbalanced rotor in a LifeLine file
exp_data = LifeLine_file.LifeLine_file(filename, low_cut=40,
	is_balanced=True, idx_range=[18, 73])

# Add experimental data to the training/testing dataset
exp_data.add_all_variables(X, Y)

# Read experimental data for balanced rotor in a LifeLine file
exp_data2 = LifeLine_file.LifeLine_file(filename, low_cut=40,
	is_balanced=False, idx_range=[185, 240])

# Add simulated data to the training/testing dataset
exp_data2.add_all_variables(X, Y)

# Normalize the data
X = tf.keras.utils.normalize(X, axis=1)

# Convert the label (output) data list to a column vector: [m,1] array
Y = np.array(Y)[np.newaxis].T

# Shuffle the data
X, Y = shuffle(X, Y)

# Split data into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Create the model
model = Sequential()
model.add(Flatten(input_shape=(X.shape[1],)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Create the adam optimizer and specify the learning rate
adam = tf.keras.optimizers.Adam(lr=1e-4)

# Compile the model using binary crossentropy as the loss function
model.compile(optimizer=adam,
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

# Train the model and record the iteration history
history = model.fit(X_train, Y_train, epochs=200, shuffle=True,
					validation_data=(X_test, Y_test))

# Evaluate the new model
loss, accuracy = model.evaluate(X_test, Y_test)

print('Model Accuracy: ', accuracy)
print(model.summary())

# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy [x/1]')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid('on')
# plt.show()
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (Cost function value)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid('on')
plt.show()

print(history.history['accuracy'])