"""
A script that contains all of the plotting utilities needed to analyse the output
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.color_palette("cubehelix", as_cmap=True)
# the relative feature importances given by the feature relevance analysis subgroup
reference_feature_relevance = [0.164419, 0.091007, 0.061295, 0.054885, 0.052883, 0.049884, 0.047604,
                               0.046830, 0.045457, 0.045169, 0.045155, 0.044796, 0.042368, 0.040893,
                               0.031758, 0.030688, 0.030566, 0.027043, 0.026590, 0.020711]


def sample_size_plot(sample_directory, target_dir):
    """
    Reads in the result files for runs with subsequently larger training set subsamples, stores
    them in a DataFrame and creates a plot.
    Args:
        sample_directory: the directory in which the result runs were stores
        target_dir: the directory to which the plot should be saved

    Returns: a plot image
    """

    result_data = pd.DataFrame()

    # iterate through all subdirectories corresponding to a certain sample size
    for sample_size_dir in os.listdir(sample_directory):
        # create a list to store the results in
        rmse_list = []
        pearsonr_list = []
        spearmanr_list = []
        lml_list = []

        # iterate through all the runs of a certain sample size
        sample_size = re.findall('sample_size_(.*)', sample_size_dir)[0]
        for run in os.listdir(sample_directory + '/' + sample_size_dir):

            # open all files ending in .csv (the one result file) and store the performance metrics
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

        # append the performance metrics to the dataframe, annotated with the sample size
        tuples = list(zip(([int(sample_size)] * len(lml_list)), lml_list, rmse_list, pearsonr_list, spearmanr_list))
        result_data = result_data.append(tuples, ignore_index=True)

    result_data.columns = ['sample_size', 'lml', 'rmse', 'pearsonr', 'spearmanr']
    result_data = result_data[['sample_size', 'rmse', 'pearsonr', 'spearmanr']].sort_values('sample_size')

    # plot the data on a facet grid, with a lineplot for each performance metric
    plot_data = pd.melt(result_data, id_vars='sample_size', var_name='metric')
    g = sns.FacetGrid(plot_data, row='metric', sharey=False, aspect=2.1)
    g.map(sns.lineplot, 'sample_size', 'value', estimator='mean', ci=95)
    g.set_axis_labels("training size set", "performance metric")
    g.despine()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(
        f'Regression performance for randomly drawn training sets of different size.\n(n=10 for <=1500, n=5 for (1500, 3000), n=1 for >=3000, mean+95%CI)')
    g.tight_layout()
    plt.savefig(target_dir + f'/sample_size_metrics', dpi=200)
    plt.close()


def plot_feature_add(sample_directory, relevance, target_dir):
    """
    Read in the results for the runs with increasing feature numbers and plot the respective performance
    metrics in comparison to the expected changes, given by the feature selection subgroup.
    Args:
        sample_directory: the directory in which the results are stored
        relevance: the expected relevance of each feature (ordered from most to least important)
        target_dir: directory to which the output graph should be written

    Returns: an image of the output graph
    """

    result_data = pd.DataFrame()

    # iterate through all repetitions of the experiment
    for run in os.listdir(sample_directory):
        run_num = int(re.findall('adding_features_(.*)', run)[0])
        rmse_list = []
        pearsonr_list = []
        spearmanr_list = []
        lml_list = []
        num_feat_list = []

        # iterate trough all instances, with increasing feature numbers
        for num_feat_dir in os.listdir(sample_directory + '/' + run):
            num_feat = int(re.findall('number_features(.*)', num_feat_dir)[0])
            num_feat_list.append(num_feat)

            # open all files ending in .csv (the one result file) and store the performance metrics
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

        # append the performance metrics to the dataframe, annotated with the sample size
        tuples = list(zip(num_feat_list, lml_list, rmse_list, pearsonr_list, spearmanr_list))
        result_data = result_data.append(tuples, ignore_index=True)
    result_data.columns = ['sample_size', 'lml', 'rmse', 'pearsonr', 'spearmanr']
    result_data = result_data[['sample_size', 'rmse', 'pearsonr', 'spearmanr']].sort_values('sample_size')

    # convert the relevance array to a cumulative relevance array
    relevance = np.array([sum(relevance[:i + 1]) for i in range(0, len(relevance))])
    # get the mean values of the performance metrics across all instantiations
    means = result_data.groupby('sample_size').mean()

    # rescale the relevance array to be comparable to each performance array
    # i.e. set the start and end points equal and rescale the rest
    # as we only care about the relative increases
    reverse_relevance = np.flip(relevance)
    rescaled_rmse = np.interp(reverse_relevance, (reverse_relevance.min(), reverse_relevance.max()),
                              (means['rmse'].min(), means['rmse'].max())).tolist()
    rescaled_pearsonr = np.interp(relevance, (relevance.min(), relevance.max()),
                                  (means['pearsonr'].min(), means['pearsonr'].max())).tolist()
    rescaled_spearmanr = np.interp(relevance, (relevance.min(), relevance.max()),
                                   (means['spearmanr'].min(), means['spearmanr'].max())).tolist()

    # store everything in a DataFrame
    tuples = list(zip(range(1, len(relevance) + 1), rescaled_rmse, rescaled_pearsonr, rescaled_spearmanr))
    rescaled_df = pd.DataFrame(tuples, columns=['sample_size', 'rmse', 'pearsonr', 'spearmanr'])

    # reshape the DataFrames to allow for automated plotting of both
    plot_data = pd.melt(result_data, id_vars='sample_size', var_name='metric')
    plot_relevance = pd.melt(rescaled_df, id_vars='sample_size', var_name='metric')

    plot_relevance['relevance'] = ["calculated"] * len(plot_relevance.index)
    plot_data['relevance'] = ["experimental"] * len(plot_data.index)
    plot_combined = pd.concat([plot_data, plot_relevance], ignore_index=True)

    # plot the change in performance metrics for increasing feature numbers
    # and overlay the expected relative change
    g = sns.FacetGrid(plot_combined, col='metric', hue='relevance', sharey=False)
    g.map(sns.lineplot, 'sample_size', 'value', estimator='mean', ci=95)
    g.set_axis_labels("number of features", "performance metric")
    g.despine()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(f'Regression performance as a function of feature set size\nn=3 for <15, n=2 for >=15, mean+95% CI')
    g.tight_layout()
    plt.savefig(target_dir + f'/feature_add_metrics', dpi=200)
    plt.close()


def plot_heatmap(model_directory, result_directory, kernel_name, metrics):
    """
    Plots the heatmaps for different hyperparameters and evaluation metrics
    Args:
        result_data: the data extracted from the results files

    Returns: saves heatmaps as images
    """
    result_data = pd.DataFrame()

    # iterate through all of the model type subdirectories
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
                        # extract the performance metrics
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

    # an auxiliary function for plotting purposes
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    # rearrange data and plot the different parameters in a heatmap
    molten = pd.melt(result_data, id_vars=['lsm', 'nvm'], value_vars=metrics, var_name='metric')
    g = sns.FacetGrid(molten, col='metric')
    g.map_dataframe(draw_heatmap, 'lsm', 'nvm', 'value', linewidths=.5, cmap="vlag")
    g.set_axis_labels("length scale multiplier", "noise variance multiplier")
    g.despine()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f'Regression performance metrics for different {kernel_name} kernel hyperparameters, all others set to 1')
    g.tight_layout()
    plt.savefig(result_directory + f'/heatmap_{name}.png', dpi=200)
    plt.close()


def plot_feature_rankings(lengthscales, feature_names, figpath, reciprocal=True, relative=True):
    """
    Orders the inverse, normalised lengthscales and plots them in a bar chart. Larger inverse
    lengthscales mean a more predictve power for a certain feature.
    Args:
        lengthscales: an array containing the lengthscales to plot
        feature_names: an array with the feature names (in the same order as the lengthscales)
        figpath: the path to which the figure should be saved
        reciprocal: whether to consider the inverse lengthscales
        relative: whether to normalize the lengthscales

    Returns: saves a file to the specified path
    """
    rec_string = ""
    rel_string = ""
    if reciprocal:
        lengthscales = np.reciprocal(lengthscales)
        rec_string = "reciprocal "
    if relative:
        lengthscales = lengthscales / np.amax(lengthscales)
        rel_string = "normalized "
    feature_rankings = pd.DataFrame(data=lengthscales, columns=['lengthscales'], index=feature_names)
    feature_rankings = feature_rankings.sort_values(by='lengthscales', ascending=False)

    # plot the data
    fig = plt.figure()
    sns.barplot(x='lengthscales', y=feature_rankings.index, data=feature_rankings, palette="vlag")
    plt.xlabel(rec_string + rel_string + "length scales")
    fig.tight_layout()
    plt.savefig(figpath, bbox_inches='tight', dpi=150)
    plt.close()


def plot_parameter_change(parameter_log, figpath):
    """
    A function to plot the change of parameters over the course of their optimisation.
    Args:
        parameter_log: a list of lists containing the parameters at each iteration
        figpath: the path to which the file should be written

    Returns: writes a file to the specified path

    """
    # extract the parameters from the parameter log
    indices = [item[0] for item in parameter_log]
    parameters = [item[1] for item in parameter_log]
    values = []
    # convert the tensors to lists or floats, depending on their type
    for parameter in parameters:
        param = []
        for tensor in parameter:
            if isinstance(tensor.numpy(), np.ndarray):
                param.extend(tensor.numpy().tolist())
            else:
                param.append(tensor.numpy())
        values.append(param)

    # save all of the parameters to a dataframe
    data = pd.DataFrame(index=indices, data=values)

    fig = plt.figure()
    sns.color_palette("rocket_r", as_cmap=True)
    sns.lineplot(data=data)
    fig.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()
