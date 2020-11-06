from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import gpflow as gp
import numpy as np
from sklearn.preprocessing import scale

# define reduced training set
number_of_lines = 100
reduced_features = features_train.iloc[:number_of_lines]
reduced_affinity = affinity_train.iloc[:number_of_lines]

print(reduced_affinity)
print(reduced_features)

# variance estimation
signal_variance = affinity_train.var()
noise_variance = signal_variance/5

# length scale estimation
length_scale = np.sqrt(features_train.var(axis=0).values)/2

# center labels
affinity_test = scale(affinity_test, with_mean=True, with_std=False)
affinity_train = scale(affinity_train, with_mean=True, with_std=False)

# choosing a kernel
k = gp.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scale)
m = gp.models.GPR(data=(reduced_features, reduced_affinity), kernel=k, mean_function=None)
m.likelihood.variance.assign(noise_variance)
gp.utilities.print_summary(m)

opt = gp.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
gp.utilities.print_summary(m)