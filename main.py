from GP_regression.empirical_bayes_GPR import maximize_marginal_likelihood
from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
from sklearn.preprocessing import StandardScaler
import numpy as np
import gpflow as gp
import os
import itertools
normalize_features = True
center_labels = True
output_path = 'output'


def matern52_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):

    # creating output directory
    output_directory = output_path+'/matern52'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # iterating Matern52 kernel
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05]

    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            k = gp.kernels.Matern52(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         normalize_features=normalize_features, center_labels=center_labels)

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def RBF_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):

    # creating output directory
    output_directory = output_path+'/rbf'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # iterating RBF kernel
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            k = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         normalize_features=normalize_features, center_labels=center_labels)

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

def RBF_linear_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):

    # creating output directory
    output_directory = output_path+'/rbf_linear'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # iterating RBF kernel
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            rbf = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            lin = gp.kernels.Linear()
            k = rbf + lin
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         normalize_features=normalize_features, center_labels=center_labels)

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def RBF_product_linear_grid(noise_variance, signal_variance, length_scales, training_data, testing_data):

    # creating output directory
    output_directory = output_path+'/rbf_product_linear'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # iterating RBF kernel
    signal_variance_multiplier = [1]
    length_scale_multiplier = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    noise_variance_multiplier = [0.5, 0.1, 0.05, 0.01]

    for svm, lsm, nvm in itertools.product(signal_variance_multiplier, length_scale_multiplier, noise_variance_multiplier):

        instance_directory = output_directory + f'/svm_{svm}_lsm_{lsm}_nvm_{nvm}'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for svm {svm}, lsm {lsm}, nvm {nvm}.')
        else:

            rbf = gp.kernels.SquaredExponential(variance=(signal_variance*svm), lengthscales=(length_scales*lsm))
            lin = gp.kernels.Linear()
            k = rbf*lin
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance*nvm)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         normalize_features=normalize_features, center_labels=center_labels)

            print(f'Finished {k.name} kernel with svm {svm}, lsm {lsm}, nvm {nvm}.')


def polynomial_grid(noise_variance, training_data, testing_data):

    # creating output directory
    output_directory = output_path+'/poly'
    try:
        os.mkdir(output_directory)
    except OSError:
        pass

    # iterating RBF kernel
    degree = [4, 5, 6, 7, 8, 9, 10]
    variance = [0.5, 1, 2]

    for deg, var in itertools.product(degree, variance):

        instance_directory = output_directory + f'/deg_{deg}_lvar_{var}_nvm_1'
        try:
            os.mkdir(instance_directory)
        except OSError:
            print(f'Calculation already complete for deg {deg}, var {var}.')
        else:

            k = gp.kernels.Polynomial(degree=deg, variance=var)
            m = gp.models.GPR(data=(training_data['features_train'], training_data['affinity_train']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance)

            opt = gp.optimizers.Scipy()

            print(f'Starting {k.name} kernel with deg {deg}, var {var}, nvm 1.')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=instance_directory,
                                         testing_data=testing_data,
                                         normalize_features=normalize_features, center_labels=center_labels)

            print(f'Finished {k.name} kernel with deg {deg}, var {var}, nvm 1.')


# check for duplicates
if not features_train.index.intersection(features_test.index).empty:
    raise ValueError('Training and test set are not disjunct.')

if normalize_features:
    # center and normalise features (doesn't affect prediction quality, but drastically shortens runtime)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

if center_labels:
    # center labels
    aff_mean = affinity_train.mean()
    affinity_train = affinity_train - aff_mean
    affinity_test = affinity_test - aff_mean

# setting initial parameters
signal_variance = affinity_train.var()
noise_variance = signal_variance
length_scale = np.sqrt(np.var(features_train, axis=0))

# reshaping the data
affinity_train = affinity_train.values.reshape(-1, 1)

training_data = {'features_train': features_train, 'affinity_train': affinity_train}
testing_data = {'features_test': features_test, 'affinity_test': affinity_test}

polynomial_grid(noise_variance=noise_variance, training_data=training_data, testing_data=testing_data)

matern52_grid(noise_variance=noise_variance, signal_variance=signal_variance, length_scales=length_scale,
              training_data=training_data, testing_data=testing_data)


