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
    for sample_size_dir in os.listdir(sample_directory):
        rmse_list = []
        pearsonr_list = []
        spearmanr_list = []
        lml_list = []

        sample_size = re.findall('sample_size_(.*)', sample_size_dir)[0]


        for run in os.listdir(sample_directory+'/'+sample_size_dir):
            # and open the result files (ending in .csv)
            for filename in os.listdir(sample_directory + '/' + sample_size_dir + '/' + run):
                if re.match('\w+.csv', filename):
                    with open(sample_directory + '/' + sample_size_dir + '/' + run + '/' + filename) as result_file:
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

        tuples = list(zip(([int(sample_size)]*len(lml_list)), lml_list, rmse_list, pearsonr_list, spearmanr_list))
        result_data = result_data.append(tuples, ignore_index=True)

    result_data.columns = ['sample_size','lml','rmse','pearsonr','spearmanr']
    result_data = result_data[['sample_size','rmse','pearsonr','spearmanr']].sort_values('sample_size')

    return result_data


def sample_plot(data, target_dir):
    plot_data = pd.melt(data, id_vars='sample_size', var_name='metric')
    g = sns.FacetGrid(plot_data, row='metric', sharey=False, aspect=2.1)
    g.map(sns.lineplot,'sample_size','value', estimator='mean', ci=95)
    g.set_axis_labels("training size set", "performance metric")
    g.despine()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(f'Regression performance for randomly drawn training sets of different size.\n(n=10 for <=1500, n=5 for (1500, 3000), n=1 for >=3000, mean+95%CI)')
    g.tight_layout()
    plt.savefig(target_dir+f'/sample_size_metrics', dpi=200)


if __name__ == '__main__':
    res = read_in_results(output_directory+'/random_sampling')
    sample_plot(res, result_directory)