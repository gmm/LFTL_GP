import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import tensorflow as tf

import gpflow
from gpflow.ci_utils import ci_niter


def plot_feature_rankings(lengthscales, feature_names,  figpath, reciprocal=True, relative=True):
    rec_string=""
    rel_string=""
    if reciprocal:
        lengthscales = np.reciprocal(lengthscales)
        rec_string="reciprocal "
    if relative:
        lengthscales = lengthscales / np.amax(lengthscales)
        rel_string = "normalized "
    feature_rankings = pd.DataFrame(data=lengthscales, columns=['lengthscales'], index=feature_names)
    feature_rankings = feature_rankings.sort_values(by='lengthscales', ascending=False)

    # plot the data
    fig = plt.figure()
    sns.barplot(x='lengthscales', y=feature_rankings.index, data=feature_rankings, palette="vlag")
    plt.xlabel(rec_string+rel_string+"length scales")
    fig.tight_layout()
    plt.savefig(figpath, bbox_inches='tight', dpi=150)
    plt.close()


def plot_parameter_change(parameter_log, figpath):
    indices = [item[0] for item in parameter_log]
    parameters = [item[1] for item in parameter_log]
    values = []
    for parameter in parameters:
        param = []
        for tensor in parameter:
            if isinstance(tensor.numpy(), np.ndarray):
                param.extend(tensor.numpy().tolist())
            else:
                param.append(tensor.numpy())
        values.append(param)
    data = pd.DataFrame(index=indices, data=values)

    fig = plt.figure()
    sns.color_palette("rocket_r", as_cmap=True)
    sns.lineplot(data=data)
    fig.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()