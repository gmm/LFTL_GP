"""
This script contains a short demo, creating a Gaussian Process with a squared exponential + linear
sum kernel, optimising it for different training sample set sizes and plotting the change in
performance metrics for training set of increasing size.
"""

from GP_regression.feature_data_exploration import increasing_random_samples
from GP_regression.utilities import sample_size_plot

# defining the sample size range and the used features (as given by the feature selection subgroup)
sample_range = list(range(50, 501, 25))
features_by_relevance = ['vina_gauss2', 'BertzCT', 'MolLogP', 'PEOE_VSA7', 'BalabanJ', 'SMR_VSA7', 'as_flex_all',
                         'vina_hydrogen', 'MaxAbsEStateIndex', 'HallKierAlpha', 'MaxAbsPartialCharge',
                         'vina_repulsion', 'PEOE_VSA8', 'PEOE_VSA1', 'lig_OA', 'PEOE_VSA3',
                         'NOCount', 'PEOE_VSA6', 'cc_A.HD_4', 'cc_N.N_4']

output_and_result_directory = '/demo'

increasing_random_samples(sample_sizes=sample_range, feature_order=features_by_relevance,
                          output_path=output_and_result_directory)
sample_size_plot(sample_directory=output_and_result_directory, target_dir=output_and_result_directory)
