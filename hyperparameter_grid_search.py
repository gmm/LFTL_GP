"""
This script defines the functions needed to perform a grid search to find optimal hyperparameters.
The currently implemented kernels include Matern52, RBF, RBF+linear, RBF*linear and Polynomial.
All of the results are stored in appropriately named directories in the specified output location.

NOTE: Most of the functions for the RBF-derived kernels if rather similar, but still different
enough to warrant a separate implementation.
"""


from GP_regression.empirical_bayes_GPR import maximize_marginal_likelihood
import gpflow as gp
import os
import itertools
output_path = 'gridsearch_output'


def matern52_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):
    """
    A function that performs a hyperparameter grid search for the Matern52 kernel,
    as implemented in GPflow2.
    Args:
        noise_variance: the likelihood variance value around which the grid search is to be done
        signal_variance: the kernel variance value around which the grid search is to be done
        length_scales: the kernel lengthscale value around which the grid search is to be done
        training_data: the data on which to train the model
        testing_data: the data on which to test the model

    Returns: automatically creates the appropriate output directories into which the result is stored
    """

    # check whether main output directory exists/can be created
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # check whether kernel-specific output directory exists/can be created
    output_directory = output_path+'/matern52'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # specify multipliers for the different parameter values at which to initialise the model
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05]

    # iterate through all permutations of the above-specified multipliers
    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        # create an instance directory for the specified parameter pair
        # skip if it already exists (e.g. during a re-run with previously evaluated param pairs)
        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            # specify the respective model, kernel and optimiser and run the optimisation
            k = gp.kernels.Matern52(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         plot_params=False, plot_ARD=True)
            # NOTE: make sure to check model.kernel.lenghtscales
            # and not model.kernel.kernels[0].lengthscales (for composite kernels) is selected in the GPR script

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def rbf_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):
    """
    A function that performs a hyperparameter grid search for the RBF kernel,
    as implemented in GPflow2.
    Args:
        noise_variance: the likelihood variance value around which the grid search is to be done
        signal_variance: the kernel variance value around which the grid search is to be done
        length_scales: the kernel lengthscale value around which the grid search is to be done
        training_data: the data on which to train the model
        testing_data: the data on which to test the model

    Returns: automatically creates the appropriate output directories into which the result is stored
    """

    # check whether main output directory exists/can be created
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # check whether kernel-specific output directory exists/can be created
    output_directory = output_path+'/rbf'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # specify multipliers for the different parameter values at which to initialise the model
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    # iterate through all permutations of the above-specified multipliers
    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        # create an instance directory for the specified parameter pair
        # skip if it already exists (e.g. during a re-run with previously evaluated param pairs)
        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            # specify the respective model, kernel and optimiser and run the optimisation
            k = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         plot_params=False, plot_ARD=True)
            # NOTE: make sure to check model.kernel.lenghtscales
            # and not model.kernel.kernels[0].lengthscales (for composite kernels) is selected in the GPR script

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def rbf_linear_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):
    """
    A function that performs a hyperparameter grid search for the sum of the RBF and linear kernel,
    as implemented in GPflow2.
    Args:
        noise_variance: the likelihood variance value around which the grid search is to be done
        signal_variance: the kernel variance value around which the grid search is to be done
        length_scales: the kernel lengthscale value around which the grid search is to be done
        training_data: the data on which to train the model
        testing_data: the data on which to test the model

    Returns: automatically creates the appropriate output directories into which the result is stored
    """

    # check whether main output directory exists/can be created
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # check whether kernel-specific output directory exists/can be created
    output_directory = output_path+'/rbf_linear'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # specify multipliers for the different parameter values at which to initialise the model
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    # iterate through all permutations of the above-specified multipliers
    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        # create an instance directory for the specified parameter pair
        # skip if it already exists (e.g. during a re-run with previously evaluated param pairs)
        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            # specify the respective model, kernel and optimiser and run the optimisation
            rbf = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            lin = gp.kernels.Linear()
            k = rbf + lin
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         plot_params=False, plot_ARD=True)
            # NOTE: make sure to check model.kernel.kernels[0].lengthscales
            # and not model.kernel.lenghtscales (for composite kernels) is selected in the GPR script

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def rbf_product_linear_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):
    """
    A function that performs a hyperparameter grid search for the product of the RBF and linear kernel,
    as implemented in GPflow2.
    Args:
        noise_variance: the likelihood variance value around which the grid search is to be done
        signal_variance: the kernel variance value around which the grid search is to be done
        length_scales: the kernel lengthscale value around which the grid search is to be done
        training_data: the data on which to train the model
        testing_data: the data on which to test the model

    Returns: automatically creates the appropriate output directories into which the result is stored
    """

    # check whether main output directory exists/can be created
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # check whether kernel-specific output directory exists/can be created
    output_directory = output_path+'/rbf_product_linear'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # specify multipliers for the different parameter values at which to initialise the model
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    # iterate through all permutations of the above-specified multipliers
    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        # create an instance directory for the specified parameter pair
        # skip if it already exists (e.g. during a re-run with previously evaluated param pairs)
        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            # specify the respective model, kernel and optimiser and run the optimisation
            rbf = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            lin = gp.kernels.Linear()
            k = rbf*lin
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         plot_params=False, plot_ARD=True)
            # NOTE: make sure to check model.kernel.kernels[0].lengthscales
            # and not model.kernel.lenghtscales (for composite kernels) is selected in the GPR script

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def polynomial_grid(noise_variance, training_data, testing_data):
    """
    A function that performs a hyperparameter grid search for the polynomial kernel,
    as implemented in GPflow2.
    Args:
        noise_variance: the likelihood variance value around which the grid search is to be done
        signal_variance: the kernel variance value around which the grid search is to be done
        length_scales: the kernel lengthscale value around which the grid search is to be done
        training_data: the data on which to train the model
        testing_data: the data on which to test the model

    Returns: automatically creates the appropriate output directories into which the result is stored
    """

    # check whether main output directory exists/can be created
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # check whether kernel-specific output directory exists/can be created
    output_directory = output_path+'/poly'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # specify values for the different parameter values at which to initialise the model
    degree = [4, 5, 6, 7, 8, 9, 10]
    variance = [0.5, 1, 2]

    # iterate through all permutations of the above-specified values
    for deg, var in itertools.product(degree, variance):

        # create an instance directory for the specified parameter pair
        # skip if it already exists (e.g. during a re-run with previously evaluated param pairs)
        instance_directory = output_directory + f'/deg_{deg}_lvar_{var}_nvm_1'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for deg {deg}, var {var}.')
        else:

            # specify the respective model, kernel and optimiser and run the optimisation
            k = gp.kernels.Polynomial(degree=deg, variance=var)
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with deg {deg}, var {var}, nvm 1.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         plot_params=False, plot_ARD=False)

            print(f'Finished {k.name} kernel with deg {deg}, var {var}, nvm 1.')
