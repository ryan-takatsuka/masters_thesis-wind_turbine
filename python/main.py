# Import stuff
import split_dft
import LifeLine_file
import simulate_LL_data
import detectionModel
import matplotlib.pyplot as plt
import numpy as np


# Read experimental data for an imbalanced rotor in a LifeLine file
exp_data = LifeLine_file.LifeLine_file('WDFT_2018-04-01_160352.txt', low_cut=42,
	is_balanced=True, idx_range=[18, 73])
# exp_data.filter(ftype='SG', order=2, framelen=15, wn=0.1)
exp_data.filter(ftype='Butter_IIR', order=2, framelen=7, wn=0.2)
# exp_data.filter(ftype='Window_FIR', order=8, framelen=7, wn=0.2)
exp_data.plot()

# Add experimental data to the training/testing dataset
dataset = []
exp_data.addto_dataset(dataset)

# Create simulated data for a rotor imbalance (because I don't have this data)
# sim_data = simulate_LL_data.simulated_data(exp_data.time)
# sim_data.simulate_imbalance()

# Read experimental data for balanced rotor in a LifeLine file
exp_data2 = LifeLine_file.LifeLine_file('WDFT_2018-04-01_160352.txt', low_cut=42,
	is_balanced=False, idx_range=[185, 240])
# exp_data2.filter(ftype='SG', order=2, framelen=15, wn=0.1)
exp_data2.filter(ftype='Butter_IIR', order=2, framelen=7, wn=0.2)
# exp_data2.filter(ftype='Window_FIR', order=8, framelen=7, wn=0.2)
exp_data2.plot()

# Add simulated data to the training/testing dataset
exp_data2.addto_dataset(dataset)

# Create the detection model for determining an imbalance in the rotor
my_algorithm = detectionModel.detectionModel(dataset)

# Test different types of models to see which one performs the best
my_algorithm.test_different_models(plot=True)

# Train and validate the model
model = my_algorithm.train()
predictions = my_algorithm.test()

# Plot model
my_algorithm.plot_model()


# plt.show()


print(np.array(exp_data2.amplitudes).shape)


print(my_algorithm.X_train.shape)
print(my_algorithm.X_test.shape)