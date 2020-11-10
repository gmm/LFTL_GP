import matplotlib.pyplot as plt
from GP_regression.data_extraction import features_data
import numpy as np
import seaborn as sns
import pandas as pd


def plot_feature_rankings():
    lengthscales = np.reciprocal(np.array([0.5635456372420777, 0.1770943592142015, 0.014031030017897456, 0.2923324238121054, 0.010423891672246806, 0.010551633455491873, 0.19665380428839577, 0.0417655020025933, 0.0010556729903651739, 0.3772712292653649, 0.09059733230061488, 0.8608453382428183]))
    lengthscales = lengthscales / np.amax(lengthscales)
    feature_rankings = pd.DataFrame(data=lengthscales, columns=['lengthscales'], index=features_data.columns)
    feature_rankings = feature_rankings.sort_values(by='lengthscales', ascending=False)
    sns.barplot(x='lengthscales', y=feature_rankings.index, data=feature_rankings, palette="vlag")
    plt.tight_layout()
    plt.xlabel("Normalised, reciprocal length-scale")
    plt.show()

plot_feature_rankings()