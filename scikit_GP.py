from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF, WhiteKernel, ExpSineSquared, RationalQuadratic, Exponential, WhiteKernel, DotProduct
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time 



if not features_train.index.intersection(features_test.index).empty:
    raise ValueError('Training and test set are not disjunct.')


# The features have different scales so we standardise
scaler = StandardScaler()
features_train  = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# variance estimation
signal_variance = affinity_train.var()
noise_variance = signal_variance/5

# length scale estimation
length_scale = np.sqrt(features_train.var(axis=0))/2



t = time.process_time()

# Choose a kernel  
k = Matern(length_scale = np.sqrt(features_train.var(axis=0))/2 , nu = 2.5) 

# Run Gaussian Processes regression 
gpr = GaussianProcessRegressor(kernel=k, optimizer = 'fmin_l_bfgs_b',alpha = 1, n_restarts_optimizer=3) 


# print parameters
print(gpr.get_params())


# Fit to data using Maximum Likelihood Estimation of the parameters
gpr.fit(features_train, affinity_train) 


# Make the prediction on the testing set

y_pred_test, sigma = gpr.predict(features_test, return_std=True)

elapsed_time = time.process_time() - t

RMSE = np.sqrt(mean_squared_error(affinity_test,y_pred_test))

R2 = r2_score(affinity_test, y_pred_test)

pearsonsr = pearsonr(affinity_test,y_pred_test)[0]

print('elapsed time', elapsed_time)

print("RMSE", RMSE)
 
print("R2", R2)

print("Pearson's", pearsonsr)

###Learning curves### 


def learning_curves(estimator, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(estimator, features, target, train_sizes ,cv = cv, 
    scoring = 'neg_mean_squared_error')
    train_scores_mean = np.sqrt(-train_scores.mean(axis = 1))
    validation_scores_mean = np.sqrt(-validation_scores.mean(axis = 1))
    train_scores_std = np.std(-train_scores.mean(axis = 1))
    validation_scores_std = np.std(-validation_scores.mean(axis = 1))
    
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                         validation_scores_mean + validation_scores_std, alpha=0.1,
                         color="g")

    plt.ylabel('rMSE', fontsize = 18)
    plt.xlabel('Training set size', fontsize = 18)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.rcParams.update({'font.size': 18})
    plt.yticks(np.arange(0,3,0.2))

### Plot the learning curves ###
plt.figure(figsize = (10,7))
train_sizes = [500, 1000, 3000]
learning_curves(gpr, features_train, affinity_train, train_sizes, 5)

# Get the kernel parameters 
gpr.kernel_.get_params()


### Hyperparameter optimisation using Manual Tuning ###


RBF_alpha = [0,0.5,1,2,3,5]
RBF_rho = [0.56,0.58,0.68,0.65,0.59,0.59]

fig, ax = plt.subplots()
ax.scatter(alpha,rho)
ax.set_ylim([0.5,0.7])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\rho$')
plt.rcParams.update({'font.size': 16})
plt.savefig('Correlation_changing_alpha_RBF.png')
plt.show()


### Hyperparameter optimisation using Grid Search ##


t = time.process_time()
param_grid = {'kernel__length_scale': [0.25,0.3,0.5,1,10],
              "kernel": [RBF(), RBF() + ConstantKernel(),RBF()*ConstantKernel,RBF() + DotProduct(), RBF() + Exponentiation],
              "alpha": [0,0.1,1,2],
              "optimizer": ["fmin_l_bfgs_b"],
              "n_restarts_optimizer": [2,3,5],
              "normalize_y": [False],
              "copy_X_train": [True], 
              "random_state": [0]}

print("\nRunning grid search to tune GPR parameters")
gpr = GaussianProcessRegressor()
grid_search = GridSearchCV(gpr, param_grid=param_grid,n_jobs=-1)
grid_search = grid_search.fit(features_train, affinity_train)

# Print best parameters
print("\nBest parameters from grid search", grid_search.best_params_)

# Use the best estimator across all search
gp_optimised = grid_search.best_estimator_

# Using the optimised GPR, make prediction on the testing set

y_pred_test = gp_optimised.predict(features_test)

elapsed_time = time.process_time() - t

RMSE = np.sqrt(mean_squared_error(affinity_test,y_pred_test))
R2 = r2_score(affinity_test, y_pred_test)
pearsonsr = pearsonr(affinity_test,y_pred_test)[0]

print('elapsed time', elapsed_time)
print("RMSE", np.sqrt(mean_squared_error(affinity_test,y_pred_test))) 
print("R2", r2_score(affinity_test, y_pred_test))
print("Pearson's", pearsonr(affinity_test,y_pred_test))

# Scatter plot of predicted vs observed pK affinity


f = plt.figure(figsize=(10, 8))
f.tight_layout()

plt.plot(affinity_test,y_pred_test,'o', label = r'$\rho$=0.65')
m, b = np.polyfit(affinity_test, y_pred_test, 1)
plt.plot(affinity_test, m*affinity_test + b)
plt.xlabel('Observed pK affinity (pK units)',fontsize=14)
plt.ylabel('Predicted pK affinity (pK units)',fontsize=14)
plt.legend(loc='upper left',fontsize=14)

#Plot of predictions with error bars

f, ax = plt.subplots(figsize=(7, 5))

n = 20
rng = range(n)
ax.scatter(rng, y_pred_test[:n])
ax.errorbar(rng, y_pred_test[:n], yerr=1.96*sigma[:n])

ax.set_title("Predictions with Error Bars")

ax.set_xlim((-1, 21));

### Feature relevance ###


r = permutation_importance(grid_search, features_test, affinity_test, n_repeats=30, random_state = 0)
features=[]
scores=[]
error=[]
for i in r.importances_mean.argsort()[::-1]:
  if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features_train.columns.values[i]:<8}"  f"{r.importances_mean[i]:.3f}" f" +/- {r.importances_std[i]:.3f}")
        features.append(features_train.columns.values[i])
        scores.append(r.importances_mean[i])
        error.append(r.importances_std[i])

fig, ax = plt.subplots(figsize=(15,12))
ax.bar( np.arange(len(features)), scores , error,align='center',alpha=1,ecolor='black',capsize=10)
ax.set_xlabel('Features')
ax.set_xticks(np.arange(len(features)))
ax.set_xticklabels(features)
ax.set_title('Feature Importance scores')
ax.yaxis.grid(True)
plt.tight_layout()


# plot feature relevance
plt.bar([x for x in range(len(scores))], scores)
plt.savefig('Feature_relevance_GPmodel_RBF.png')
plt.show()

# Save results in csv files

with open('output/RBF_scikit.csv', 'w') as out_file:
    out_file.write(f'Optimised GP regression RBF kernel, standardised features\n')
    out_file.write(f'Pearsons_correlation_coefficient:{pearsonsr:.4f}\n')
    out_file.write(f'R2_score:{R2:.4f}\n')
    out_file.write(f'RMSE:{RMSE}\n')
    out_file.write(f'best_parameters:{gpr.get_params()}\n')
    out_file.write(f'elapsed time: {elapsed_time}\n')
    out_file.write(f'name,y_pred_mean,sigma,y_true,{",".join(features_test.columns.to_list())}\n')
    for i in range(0, len(y_pred_test)):
        out_file.write(f'{features_test.iloc[i].name},{y_pred_test[i]:.2f},{sigma[i]:.2f},{affinity_test.iloc[i]},{str(features_test.iloc[i].to_list()).strip("[]")}\n')


