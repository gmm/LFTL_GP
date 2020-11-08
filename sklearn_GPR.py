from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import gpflow as gp
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

if not features_train.index.intersection(features_test.index).empty:
    raise ValueError('Training and test set are not disjunct.')


# center and normalise features

feat_mean = features_train.mean()
feat_var = features_train.var()
features_train = (features_train.sub(feat_mean)).div(feat_var)
features_test = (features_test.sub(feat_mean)).div(feat_var)

# center labels

aff_mean = affinity_train.mean()
affinity_train = affinity_train - aff_mean
affinity_test = affinity_test - aff_mean


# reshaping
#number_of_lines = 6
features_train = features_train.values
affinity_train = affinity_train.values.reshape(-1, 1)


features_test = features_test.values
affinity_test = affinity_test.values.reshape(-1, 1)


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
gpr.fit(features_train, affinity_train)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gpr.predict(features_train,return_std=True)

# RMSE
print(np.sqrt(mean_squared_error(affinity_train,y_pred)))
#R-squared 
print(r2_score(affinity_train,y_pred))

# Make prediction on the testing set 
y_pred_test= gpr.predict(features_test)

print(np.sqrt(mean_squared_error(affinity_test,y_pred_test))) 
print(r2_score(affinity_test, y_pred_test))


gpr.kernel_.get_params()
