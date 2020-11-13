from GP_regression.empirical_bayes_GPR import maximize_marginal_likelihood
from GP_regression.data_extraction import extract_data
from sklearn.utils import shuffle
import numpy as np
import gpflow as gp
import os
import tensorflow
import itertools
normalize_features = True
center_labels = True
output_path = 'interpretation_output'

features_by_relevance = ['vina_gauss2', 'BertzCT', 'MolLogP', 'PEOE_VSA7', 'BalabanJ', 'SMR_VSA7', 'as_flex_all',
                         'vina_hydrogen', 'MaxAbsEStateIndex', 'HallKierAlpha', 'MaxAbsPartialCharge',
                         'vina_repulsion', 'PEOE_VSA8', 'PEOE_VSA1', 'lig_OA', 'PEOE_VSA3',
                         'NOCount', 'PEOE_VSA6', 'cc_A.HD_4', 'cc_N.N_4']

sample_range = list(range(50, 1501, 25)) + list(range(1600, 3800, 100))


def add_feature_by_feature(feature_order, outdir, lsm=0.2, nvm=0.2):

    index_set = []

    add_feat_dir = outdir
    try:
        os.mkdir(add_feat_dir)
    except OSError:
        pass

    for feature in feature_order:

        print('Proceeding with feature '+feature)

        index_set.append(feature)
        step_directory = add_feat_dir + f'/number_features{len(index_set)}'

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

            # defining models
            rbf = gp.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scales)
            lin = gp.kernels.Linear()
            k = rbf + lin
            m = gp.models.GPR(data=(train['features'], train['affinity']), kernel=k, mean_function=None)
            m.likelihood.variance.assign(noise_variance)

            opt = gp.optimizers.Scipy()

            print(f'Running feature set length {len(index_set)}')

            maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=step_directory,
                                         testing_data=test, feature_names=col_order,
                                         normalize_features=normalize_features, center_labels=center_labels)


def random_samples(sample_sizes):

    sample_dir = output_path + '/random_sampling'
    try:
        os.mkdir(sample_dir)
    except OSError:
        pass

    # setting initial parameters
    signal_variance = 1
    noise_variance = np.float64(0.2)
    length_scales = np.ones(len(features_by_relevance)) * 0.2

    # defining kernel
    rbf = gp.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scales)
    lin = gp.kernels.Linear()
    k = rbf + lin

    # initializing model
    for sample_size in sample_sizes:

        size_dir = sample_dir + f'/sample_size_{sample_size}'
        try:
            os.mkdir(size_dir)
        except OSError:
            pass

        if sample_size <= 1500:
            repeats = 10
        elif sample_size > 1500 and sample_size < 3000:
            repeats = 5
        else:
            repeats = 1

        for i in range(0, repeats):

            run_dir = size_dir + f'/run_{i}'
            try:
                os.mkdir(run_dir)
            except OSError:
                print(f"Run {i} for sample size {sample_size} already exists.")
            else:
                train, test, col_order = extract_data(features_by_relevance, sample_size)

                m = gp.models.GPR(data=(train['features'], train['affinity']), kernel=k, mean_function=None)
                m.likelihood.variance.assign(noise_variance)

                opt = gp.optimizers.Scipy()

                print(f'Running sample size {sample_size}, repetition {i}')

                maximize_marginal_likelihood(kernel=k, model=m, optimizer=opt, output_directory=run_dir,
                                             testing_data=test, feature_names=col_order,
                                             normalize_features=normalize_features, center_labels=center_labels)


random_samples(sample_range)
add_feature_by_feature(feature_order=features_by_relevance, outdir=output_path + '/adding_features_1', lsm=0.5, nvm=0.15)
add_feature_by_feature(feature_order=features_by_relevance, outdir=output_path + '/adding_features_2', lsm=0.1, nvm=0.01)
