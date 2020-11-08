from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import gpflow as gp
import numpy as np
from sklearn.preprocessing import scale

# define reduced training set
number_of_lines = 100
reduced_features = features_train.iloc[:number_of_lines]
reduced_affinity = affinity_train.iloc[:number_of_lines]


# define reduced testing set
number_of_lines = 100
reduced_features_test = features_test[:number_of_lines]
reduced_affinity_test = affinity_test[:number_of_lines]

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






####################


from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, ExpSineSquared, RationalQuadratic
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



# Choose a kernel  

k1 = ConstantKernel(constant_value=66.0**2) * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = ConstantKernel(constant_value=2.4**2) * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)  
k3 = ConstantKernel(constant_value=0.66**2) * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = ConstantKernel(constant_value=0.18**2) * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
k = k1 + k2 + k3 + k4


gpr = GaussianProcessRegressor(kernel=k, optimizer = 'fmin_l_bfgs_b',alpha = 0.1**2, n_restarts_optimizer=5)

# print parameters
print(gpr.get_params())


# Fit to data using Maximum Likelihood Estimation of the parameters
gpr.fit(reduced_features, reduced_affinity)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gpr.predict(reduced_features,return_std=True)

# RMSE
print(np.sqrt(mean_squared_error(reduced_affinity,y_pred)))
#R-squared 
print(r2_score(reduced_affinity,y_pred))

# Make prediction on the testing set 
y_pred_test= gpr.predict(reduced_features_test)

print(np.sqrt(mean_squared_error(reduced_affinity_test,y_pred_test))) 
print(r2_score(reduced_affinity_test, y_pred_test))


gpr.kernel_.get_params()



