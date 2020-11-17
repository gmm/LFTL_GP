import gpflow as gp
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from GP_regression.utilities import plot_feature_rankings, plot_parameter_change
from gpflow.ci_utils import ci_niter
import numpy as np
np.set_printoptions(linewidth=np.inf)  # keep long length scale arrays in one line when writing to file


def maximize_marginal_likelihood(kernel, model, optimizer, output_directory,
                                 testing_data, feature_names, plot_ARD, plot_params):
    """
    Optimise the given model and write the predictions and resulting metrics to a file
    Args:
        kernel: the kernel object serving as the covariance function
        model: the model object to specify and fit the GP
        optimizer: the optimizer to minimise the negative log marginal likelihood
        output_directory: the directory to which the output file (and optionally plots) are written
        testing_data: the features and label on which the regression is to be evaluated
        feature_names: the names of the features used to train this model
        plot_ARD: create a graph of the feature-specific length scales
        plot_params: create a graph of how the parameters change over the course of the optimisation

    Returns: prints the results to a file in the specified directory
    """

    # run the optimiser with a callback function if user wants to track the parameters
    if plot_params:
        parameter_log = []
        opt_logs = optimizer.minimize(closure=model.training_loss, variables=model.trainable_variables,
                                      step_callback=(lambda x,y,z: parameter_log.append([x, z])),
                                      options=dict(maxiter=ci_niter(250)))

    else:
        # run the optimiser without a callback function otherwise
        opt_logs = optimizer.minimize(closure=model.training_loss, variables=model.trainable_variables,
                                      options=dict(maxiter=ci_niter(250)))

    # set data against which to validate the model
    features_test = testing_data['features']
    affinity_test = testing_data['affinity']

    # calculate the predictions and Pearson's R, Spearman's R as well as RMSE
    mean, var = model.predict_f(features_test.values)
    pearsonsr, pvalue = pearsonr(mean.numpy().flatten(), affinity_test.values)
    spearmansr, spvalue = spearmanr(a=mean.numpy().flatten(), b=affinity_test.values)
    rmse = np.sqrt(mean_squared_error(affinity_test.values, mean.numpy().flatten()))

    # write the results to a file
    filename = f'{model.name}_{kernel.name}'+'.csv'

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

    # create the plots that were specified in the arguments to the specified output directory
    if plot_ARD:
        plot_feature_rankings(lengthscales=model.kernel.kernels[0].lengthscales.numpy(),
                              feature_names=feature_names, figpath=output_directory+'/feature_relevance.png')
    if plot_params:
        plot_parameter_change(parameter_log=parameter_log, figpath=output_directory+'/parameter_change.png')
