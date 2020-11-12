import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("cubehelix", as_cmap=True)
output_directory = 'output'
result_directory = 'results'

kernel_dirs = os.listdir(output_directory)


def read_in_results(model_directory):
    """
    Reads in the results for all runs with a given kernel and a given model.
    Args:
        model_directory: Path to the directory from which data should be extracted,
        e.g. 'output/rbf'

    Returns: a dataframe with initial hyperparameters and the results after optimization
    """
    result_data = pd.DataFrame()
    runs = os.listdir(model_directory)

    # iterate through all subdirectories
    for run in os.listdir(model_directory):
        run_data = {}

        # extract the hyperparameters from the directory names
        svm = re.findall('svm_(.+?)_', run)
        lsm = re.findall('lsm_(.+?)_', run)
        nvm = re.findall('nvm_(.*)', run)

        run_data['svm'] = float(svm[0])
        run_data['lsm'] = float(lsm[0])
        run_data['nvm'] = float(nvm[0])

        # iterate through all files in the subdirectory
        for filename in os.listdir(model_directory + '/' + run):
            # and open the result files (ending in .csv)
            if re.match('\w+.csv', filename):
                with open(model_directory + '/' + run + '/' + filename) as result_file:
                    for line in result_file:
                        # read in the optimized hyperparameters
                        if line.startswith('%.'):
                            parameter, value = line.split(maxsplit=1)
                            try:
                                # if read in value can be converted to float, convert
                                run_data[parameter.strip("%.:")] = float(value.strip('\n'))
                            except ValueError:
                                # otherwise convert to a list
                                run_data[parameter.strip("%.:")] = [float(i) for i in list(value.strip('[]\n').split())]
                        # extract the performance metrics (a bit heterogenous)
                        if line.startswith("%loglikelihood"):
                            _, lml = line.split()
                            run_data['lml'] = float(lml)
                        if line.startswith("%RMSE"):
                            _, rmse = line.split(":")
                            run_data['rmse'] = float(rmse.strip('\n'))
                        if line.startswith("%Pearson_correlation"):
                            pearsonr = re.findall(':(\d\.\d+),', line)
                            run_data['pearsonr'] = float(pearsonr[0])
                        if line.startswith("%Spearman_correlation"):
                            spearmanr = re.findall(':(\d\.\d+),', line)
                            run_data['spearmanr'] = float(spearmanr[0])

        # append the data to the results dataframe
        result_data = result_data.append(run_data, ignore_index=True)

    return result_data


def plot_heatmap(result_data, metrics, kernel_name):
    """
    Plots the heatmaps for different hyperparameters and evaluation metrics
    Args:
        result_data: the data extracted from the results files

    Returns: saves heatmaps as images
    """

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    molten = pd.melt(result_data, id_vars=['lsm', 'nvm'], value_vars=metrics, var_name='metric')
    g = sns.FacetGrid(molten, col='metric')
    g.map_dataframe(draw_heatmap, 'lsm', 'nvm', 'value', linewidths=.5, cmap="vlag")
    g.set_axis_labels("length scale multiplier", "noise variance multiplier")
    g.despine()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Regression performance metrics for different {kernel_name} kernel hyperparameters, all others set to 1')
    g.tight_layout()

    plt.savefig(result_directory+f'/heatmap_{name}.png', dpi=200)


rbf_names = ['rbf', 'rbf_linear', 'rbf_product_linear']
rbf_metrics = ['pearsonr', 'spearmanr', 'lml', 'rmse']
for name in rbf_names:
    result = read_in_results(output_directory+'/'+name)
    plot_heatmap(result, rbf_metrics, name)






