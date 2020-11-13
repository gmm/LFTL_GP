import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.color_palette("cubehelix", as_cmap=True)
output_directory = 'interpretation_output'
result_directory = 'results'


def read_in_results(sample_directory):
    """
    Reads in the results for all runs with a given kernel and a given model.
    Args:
        model_directory: Path to the directory from which data should be extracted,
        e.g. 'gridsearch_output/rbf'

    Returns: a dataframe with initial hyperparameters and the results after optimization
    """
    result_data = pd.DataFrame()

    # iterate through all subdirectories
    for run in os.listdir(sample_directory):
        run_num = int(re.findall('adding_features_(.*)', run)[0])
        rmse_list = []
        pearsonr_list = []
        spearmanr_list = []
        lml_list = []
        num_feat_list = []
        for num_feat_dir in os.listdir(sample_directory+'/'+run):
            num_feat = int(re.findall('number_features(.*)', num_feat_dir)[0])
            num_feat_list.append(num_feat)
            for filename in os.listdir(sample_directory + '/' + run + '/' + num_feat_dir):
                if re.match('\w+.csv', filename):
                    with open(sample_directory + '/' + run + '/' + num_feat_dir + '/' + filename) as result_file:
                        for line in result_file:
                            if line.startswith("%loglikelihood"):
                                _, lml = line.split()
                                lml_list.append(float(lml))
                            if line.startswith("%RMSE"):
                                _, rmse = line.split(":")
                                rmse_list.append(float(rmse.strip('\n')))
                            if line.startswith("%Pearson_correlation"):
                                pearsonr = re.findall(':(\d\.\d+),', line)
                                pearsonr_list.append(float(pearsonr[0]))
                            if line.startswith("%Spearman_correlation"):
                                spearmanr = re.findall(':(\d\.\d+),', line)
                                spearmanr_list.append(float(spearmanr[0]))

        tuples = list(zip(num_feat_list, lml_list, rmse_list, pearsonr_list, spearmanr_list))
        result_data = result_data.append(tuples, ignore_index=True)
    result_data.columns = ['sample_size', 'lml', 'rmse', 'pearsonr', 'spearmanr']
    return result_data[['sample_size','rmse','pearsonr','spearmanr']].sort_values('sample_size')

def plot_feature_add(data, target_dir):
    plot_data = pd.melt(data, id_vars='sample_size', var_name='metric')
    g = sns.FacetGrid(plot_data, col='metric', sharey=False)
    g.map(sns.lineplot, 'sample_size', 'value', estimator='mean', ci=95)
    g.set_axis_labels("number of features", "performance metric")
    g.despine()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle( f'Regression performance as a function of feature set size\nn=3 for <15, n=2 for >=15, mean+95% CI')
    g.tight_layout()
    plt.savefig(target_dir + f'/feature_add_metrics', dpi=200)

if __name__ == '__main__':
    results = read_in_results(output_directory+'/'+'adding_features_all')
    plot_feature_add(results, result_directory)
