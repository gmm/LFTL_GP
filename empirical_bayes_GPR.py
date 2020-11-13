import gpflow as gp
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from GP_regression.utilities import *
from gpflow.ci_utils import ci_niter
np.set_printoptions(linewidth=np.inf)  # keep long length scale arrays in one line when writing to file



def maximize_marginal_likelihood(kernel, model, optimizer, output_directory, testing_data, feature_names, normalize_features, center_labels):
    """
    Set up a model with the specified kernel and parameters and maximize the marginal likelihood
    Args:
        kernel: the kernel object serving as the covariance function
        model: the model object to specify and fit the GP
        optimizer: the optimizer to maximize the marginal likelihood
        output_directory: the directory to which the gridsearch_output file is written
        testing_data: the features and label on which the regression is to be evaluated
        feature_names: the names of the features used to train this model
        normalize_features: whether the features should be scaled to mean 0 and variance 1
        center_labels: whether the labels should be centered around zero

    Returns: prints the results to a file in the specified directory
    """

    # non-convex optimization of the hyperparameters to maximize marginal likelihood
    # parameter_log = []
    # ad step_callback=(lambda x,y,z: parameter_log.append([x, z])) to optimizer to create callback log

    opt_logs = optimizer.minimize(closure=model.training_loss, variables=model.trainable_variables, options=dict(maxiter=ci_niter(250)))

    # set data against which to validate the model
    features_test = testing_data['features']
    affinity_test = testing_data['affinity']

# choosing a kernel
k1 = gp.kernels.Matern52(variance=signal_variance, lengthscales=length_scale)
k2 = gp.kernels.Linear(variance=signal_variance)

k = k1  + k2

m = gp.models.GPR(data=(features_train, affinity_train), kernel=k, mean_function=None)
m.likelihood.variance.assign(noise_variance)

    # create output summary file

    if center_labels:
        censtr="_ctrlbs"
    else:
        censtr=""

    if normalize_features:
        norstr="_nrmfts"
    else:
        norstr=""

    filename = f'{model.name}_{kernel.name}'+censtr+norstr+'.csv'

    with open(output_directory+'/'+filename, 'w') as out_file:
        out_file.write(f'%Gaussian process regression with a Gaussian likelihood\n')
        out_file.write(f'%model: {model.name}, kernel: {kernel.name}\n')
        out_file.write(f'Optimization success: {opt_logs.get("success")} in {opt_logs.get("nit")} iterations, {opt_logs.get("message")}\n')
        for key, value in gp.utilities.read_values(model).items():
            out_file.write(f'%{key}: {value}\n')
        out_file.write(f'%loglikelihood: {model.log_marginal_likelihood()}\n')
        out_file.write(f'%RMSE:{rmse:.3f}\n')
        out_file.write(f'%Pearson_correlation_coefficient:{pearsonsr:.3f},P-value:{pvalue:.3f}\n')
        out_file.write(f'%Spearman_correlation_coefficient:{spearmansr:.3f},P-value:{spvalue:.3f}\n')
        out_file.write('%%%%%%PREDICTIONS%%%%%\n')
        out_file.write(f'name,f_pred_mean,f_pred_var,y_true,{",".join(feature_names)}\n')
        for i in range(0, len(mean)):
            out_file.write(f'{affinity_test.index.values[i]},{mean.numpy()[i][0]:.4f},{var.numpy()[i][0]:.4f},{affinity_test.values[i]:.4f},{"".join(str(i)+"," for i in features_test[i].round(4).tolist())[:-1]}\n')
        out_file.close()

    # create feature ranking plot
    plot_feature_rankings(lengthscales=model.kernel.kernels[0].lengthscales.numpy(), feature_names=feature_names, figpath=output_directory+'/feature_relevance.png')
    #plot_parameter_change(parameter_log=parameter_log, figpath=output_directory+'/parameter_change.png')