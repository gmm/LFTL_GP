import matplotlib.pyplot as plt
from GP_regression.data_extraction import features_data
import numpy as np
import seaborn as sns
import pandas as pd


def plot_feature_rankings():
    lengthscales = np.reciprocal(np.array([0.0009625500038435711, 0.16829756916372968, 0.014236495657236245, 0.32643597907738564, 0.01274348487299226, 0.00908565156167531, 0.19299571173475982, 0.042876681737993465, 0.0010468041605374342, 0.4250317841535894, 0.058636042952652505, 0.6423355830152085]))
    lengthscales = lengthscales / np.amax(lengthscales)
    feature_rankings = pd.DataFrame(data=lengthscales, columns=['lengthscales'], index=features_data.columns)
    feature_rankings = feature_rankings.sort_values(by='lengthscales', ascending=False)
    sns.barplot(x='lengthscales', y=feature_rankings.index, data=feature_rankings, palette="vlag")
    plt.tight_layout()
    plt.xlabel("Normalised, reciprocal length-scale")
    plt.show()

plot_feature_rankings()