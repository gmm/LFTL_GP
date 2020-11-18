"""
A script containing the functions to evaluate the models with an increasing randomly sampled
subset of the training data or an increasing number of features. The only implemented model
architecture is the sum of an RBF and a linear kernel for now.
"""

from GP_regression.empirical_bayes_GPR import maximize_marginal_likelihood
from GP_regression.data_extraction import extract_data
import numpy as np
import gpflow as gp
import os


def add_feature_by_feature(feature_order, outdir, lsm, nvm):
    """
    Initialises the model with the given hyperparameter multipliers and optimises them
    for an increasing number of features, in the specified order
    Args:
        feature_order: the order in which to add the features
        outdir: the directory in which the output directories are to be created
        lsm: the length scale multiplier
        nvm: the noise variance multiplier

    Returns: creates a new directory for each instance in which the result file and (optional)
             plots are saved
    """

    index_set = []

    # make sure the output directory exists, create it if not
    try:
        os.mkdir(outdir)
    except OSError:
        pass

    for feature in feature_order:

        print('Proceeding with feature '+feature)

        # create an output directory for each feature number
        index_set.append(feature)
        step_directory = outdir + f'/number_features{len(index_set)}'

        try:
           os.mkdir(step_directory)
        except OSError:
            print(f'The run for {len(index_set)} already exists.')
        else:
            train, test, col_order = extract_data(index_set)

            # setting initial parameters
            signal_variance = 1
            noise_variance = np.float64(nvm)
            length_scales = np.ones(len(index_set)) * lsm

            # define the kernel and the model
            rbf = gp.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scales)
            lin = gp.kernels.Linear()
            k = rbf + lin
            m = gp.models.GPR(data=(train['features'], train['affinity']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance)

            opt = gp.optimizers.Scipy()

            print(f'Running feature set length {len(index_set)}')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=step_directory,
                                         testing_data=test, feature_names=col_order,
                                         plot_ARD=True, plot_params=False)


def increasing_random_samples(sample_sizes, feature_order, output_path):
    """
    Takes random samples of the training data and optimises the model on the reduced set.
    Creates an output directory for each instance in which a result file is written.
    Args:
        sample_sizes: the number of data points to sample
        feature_order: the order of the features by relevance

    Returns: a directory with the relevant subdirectories and result files
    """

    # create the relevant output subdirectory
    sample_dir = output_path + '/random_sampling'
    try:
        os.mkdir(sample_dir)
    except OSError:
        pass

    # set initial hyperparameters
    signal_variance = 1
    noise_variance = np.float64(0.4)
    length_scales = np.ones(len(feature_order)) * 0.2

    # define the kernel
    rbf = gp.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scales)
    lin = gp.kernels.Linear()
    k = rbf + lin

    # create a subdirectory for each sample size
    for sample_size in sample_sizes:

        size_dir = sample_dir + f'/sample_size_{sample_size}'
        try:
            os.mkdir(size_dir)
        except OSError:
            pass

        # define the number of repeats for each sample size range, not that GPs scale
        # cubically in the number of training points
        if sample_size <= 1500:
            repeats = 10
        elif sample_size > 1500 and sample_size < 3000:
            repeats = 5
        else:
            repeats = 1

        # create a sub_directory for each instantiation
        for i in range(0, repeats):

            run_dir = size_dir + f'/run_{i}'
            try:
                os.mkdir(run_dir)
            except OSError:
                print(f"Run {i} for sample size {sample_size} already exists.")
            else:
                # read in the training sample
                train, test, col_order = extract_data(feature_order, sample_size)

                # initialise and solve the model
                m = gp.models.GPR(data=(train['features'], train['affinity']), kernel=k, mean_function=None)
                m.likelihood.variance.assign(noise_variance)

                opt = gp.optimizers.Scipy()

                print(f'Running sample size {sample_size}, repetition {i}')

                maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=run_dir,
                                             testing_data=test, feature_names=col_order,
                                             plot_ARD=False, plot_params=False)
