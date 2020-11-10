from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import gpflow as gp
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, WhiteKernel, ExpSineSquared, RationalQuadratic, WhiteKernel, DotProduct
from sklearn.model_selection import GridSearchCV


if not features_train.index.intersection(features_test.index).empty:
    raise ValueError('Training and test set are not disjunct.')


# Define reduced training set
number_of_lines = 100
reduced_features = features_train[:number_of_lines]
reduced_affinity = affinity_train[:number_of_lines]

# Define reduced testing set
number_of_lines = 100
reduced_features_test = features_test[:number_of_lines]
reduced_affinity_test = affinity_test[:number_of_lines]



# variance estimation
signal_variance = affinity_train.var()

# length scale estimation
length_scale = np.sqrt(features_train.var(axis=0))/2


# The features have different scales so we standardise
scaler = StandardScaler()
reduced_features  = scaler.fit_transform(reduced_features)
reduced_features_test = scaler.transform(reduced_features_test)


# Choose a kernel  

k = Matern(length_scale = np.sqrt(reduced_features.var(axis=0))/2 , nu = 2.5)  #nu=1.3; length_scale = np.sqrt(reduced_features.var(axis=0))/2



gpr = GaussianProcessRegressor(kernel=k, optimizer = 'fmin_l_bfgs_b',alpha = 0.1**2, n_restarts_optimizer=5)


# print parameters
print(gpr.get_params())


# Fit to data using Maximum Likelihood Estimation of the parameters
gpr.fit(reduced_features, reduced_affinity) 


# Make the prediction on the testing set, ask for MSE

y_pred_test, sigma = gpr.predict(reduced_features_test,return_std=True)



print("RMSE", np.sqrt(mean_squared_error(reduced_affinity_test,y_pred_test)))
 
print("R2", r2_score(reduced_affinity_test, y_pred_test))

print("Pearson's", pearsonr(reduced_affinity_test,y_pred_test))



# Get the kernel parameters 
gpr.kernel_.get_params()


"""
Perform Grid Search on the parameters to get the best estimator"

"""


param_grid = {'kernel__length_scale': list(np.arange(0.6, 0.9, 0.01)), 
              'kernel__nu': [0.5, 1.5, 2.5],
              "kernel": [Matern()],
              "alpha": [0.1**2],
              "optimizer": ["fmin_l_bfgs_b"],
              "n_restarts_optimizer": [2, 5, 10],
              "normalize_y": [False],
              "copy_X_train": [True], 
              "random_state": [0]}

print("\nRunning grid search to tune GPR parameters")
gpr = GaussianProcessRegressor()
grid_search = GridSearchCV(gpr, param_grid=param_grid,n_jobs=-1)
grid_search.fit(reduced_features, reduced_affinity)


# Print best parameters
print(grid_search.best_params_)

# Use the best estimator across all search
gp_optimised = grid_search.best_estimator_

# Using the optimised GPR, make prediction on the testing set

y_pred_test = gp_optimised.predict(reduced_features_test)

print("RMSE", np.sqrt(mean_squared_error(reduced_affinity_test,y_pred_test))) 
print("R2", r2_score(reduced_affinity_test, y_pred_test))
print("Pearson's", pearsonr(reduced_affinity_test,y_pred_test))

