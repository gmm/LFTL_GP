from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
import gpflow as gp
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

if not features_train.index.intersection(features_test.index).empty:
    raise ValueError('Training and test set are not disjunct.')

# center and normalise features (doesn't affect prediction quality, but drastically shortens runtime)
feat_mean = features_train.mean()
feat_var = features_train.var()
features_train = (features_train.sub(feat_mean)).div(feat_var)
features_test = (features_test.sub(feat_mean)).div(feat_var)

# center labels
aff_mean = affinity_train.mean()
affinity_train = affinity_train - aff_mean
affinity_test = affinity_test - aff_mean

# variance estimation
#signal_variance = affinity_train.var()
#noise_variance = signal_variance/2

# length scale estimation
#length_scale = np.sqrt(features_train.var(axis=0))/2

signal_variance = 2.2531
noise_variance = 1.3104
length_scale = [0.5635456372420777, 0.1770943592142015, 0.014031030017897456, 0.2923324238121054, 0.010423891672246806,
     0.010551633455491873, 0.19665380428839577, 0.0417655020025933, 0.0010556729903651739, 0.3772712292653649,
     0.09059733230061488, 0.8608453382428183]

# reshaping
#number_of_lines = 6
features_train = features_train.values
affinity_train = affinity_train.values.reshape(-1, 1)

# choosing a kernel
k1 = gp.kernels.Matern52(variance=signal_variance, lengthscales=length_scale)
k2 = gp.kernels.Linear(variance=signal_variance)

k = k1  + k2

m = gp.models.GPR(data=(features_train, affinity_train), kernel=k, mean_function=None)
m.likelihood.variance.assign(noise_variance)

# optimizing the hyperparameters
opt = gp.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables)

# calculate results
mean, var = m.predict_f(features_test.values)
pearsonsr, pvalue = pearsonr(mean.numpy().flatten(), affinity_test.values)
spearmansr, spvalue = spearmanr(a=mean.numpy().flatten(), b=affinity_test.values)
rmse = np.sqrt(mean_squared_error(affinity_test.values, mean.numpy().flatten()))

with open('output/GPR_f_Matern52_scaledlabels.csv', 'w') as out_file:
    out_file.write(f'%Standard GP regression with Matern52 kernel and scaled data and labels\n')
    out_file.write(f'%Pearson_correlation_coefficient:{pearsonsr:.4f},P-value:{pvalue:.4f}\n')
    out_file.write(f'%Spearman_correlation_coefficient:{spearmansr:.4f},P-value:{spvalue:.4f}\n')
    out_file.write(f'%RMSE:{rmse}\n')
    out_file.write(f'%signal_variance:{m.kernel.variance.numpy():.4f},noise_variance:{m.likelihood.variance.numpy():.4f},ARD_lengthscales:{m.kernel.lengthscales.numpy().tolist()}\n')
    out_file.write(f'name,f_pred_mean,f_pred_var,y_true,{",".join(features_test.columns.to_list())}\n')
    for i in range(0, len(mean)):
        out_file.write(f'{features_test.iloc[i].name},{mean[i][0]:.2f},{var[i][0]:.2f},{affinity_test.iloc[i]},{str(features_test.iloc[i].to_list()).strip("[]")}\n')
    out_file.close()
