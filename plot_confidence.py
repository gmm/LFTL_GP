import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("cubehelix", as_cmap=True)
output_directory = 'gridsearch_output'
result_directory = 'results'
features_by_relevance = ['vina_gauss2', 'BertzCT', 'MolLogP', 'PEOE_VSA7', 'BalabanJ', 'SMR_VSA7', 'as_flex_all',
                         'vina_hydrogen', 'MaxAbsEStateIndex', 'HallKierAlpha', 'MaxAbsPartialCharge',
                         'vina_repulsion', 'PEOE_VSA8', 'PEOE_VSA1', 'lig_OA', 'PEOE_VSA3',
                         'NOCount', 'PEOE_VSA6', 'cc_A.HD_4', 'cc_N.N_4']
kernel_dirs = os.listdir(output_directory)


def read_in_results(fpath, var_threshold=2):
    """
    Reads in the results for all runs with a given kernel and a given model.
    Args:
        model_directory: Path to the directory from which data should be extracted,
        e.g. 'gridsearch_output/rbf'

    Returns: a dataframe with initial hyperparameters and the results after optimization
    """

    rmse = 1.522
    pearsonr = 0.750
    pearsonp = 0.001

    data = pd.read_csv(fpath, comment='%', index_col='name')
    feature_variance = np.sqrt(data[features_by_relevance].var()) * var_threshold

    certain_values = {}
    uncertain_values = {}

    # set all of the values to False where the predictive variance is larger than the feature variance threshold
    for feature in features_by_relevance:
        certain_values[feature]=data[feature].loc[data.f_pred_var < feature_variance[feature]].tolist()
        uncertain_values[feature]=data[feature].loc[data.f_pred_var > feature_variance[feature]].tolist()

    print(uncertain_values['PEOE_VSA3'], certain_values['PEOE_VSA3'])

if __name__ == '__main__':
    filepath = "interpretation_output/all_feature_selection.csv"
    read_in_results(filepath, 2)