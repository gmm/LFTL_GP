import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import gpflow
import os
from gpflow.ci_utils import ci_niter
from GP_regression.data_extraction import features_train, features_test, affinity_train, affinity_test
from gpflow.utilities import print_summary
gpflow.config.set_default_float(np.float64)
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

tf.random.set_seed(1)
rng = np.random.RandomState(1)

# center and normalise features (doesn't affect prediction quality, but drastically shortens runtime)
feat_mean = features_train.mean()
feat_var = features_train.var()
features_train = (features_train.sub(feat_mean)).div(feat_var)
features_test = (features_test.sub(feat_mean)).div(feat_var)

aff_mean = affinity_train.mean()
affinity_train = affinity_train - aff_mean
affinity_test = affinity_test - aff_mean

# variance estimation
signal_variance = affinity_train.var()
noise_variance = signal_variance/2

# length scale estimation
length_scale = np.sqrt(features_train.var(axis=0))/2

# import PDBbind data
data = (features_train.values, affinity_train.values.reshape(-1, 1))

kernel = gpflow.kernels.Matern52(variance=signal_variance, lengthscales=length_scale)
model = gpflow.models.GPR(data, kernel, mean_function=None, noise_variance=noise_variance)

optimizer = gpflow.optimizers.Scipy()
maxiter = ci_niter(3000)
_ = optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=maxiter))

model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

num_burnin_steps = ci_niter(600)
num_samples = ci_niter(1500)

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)

@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

samples, traces = run_chain_fn()
parameter_samples = hmc_helper.convert_to_constrained_values(samples)

param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(model).items()}

path = 'GPR_HMC_Matern52_centredlabels'

try:
    os.mkdir(path)
except OSError:
    pass

# calculate results
mean, var = model.predict_f(features_test.values)
pearsonsr, pvalue = pearsonr(mean.numpy().flatten(), affinity_test.values)
spearmansr, spvalue = spearmanr(a=mean.numpy().flatten(), b=affinity_test.values)
rmse = np.sqrt(mean_squared_error(affinity_test.values, mean.numpy().flatten()))

with open(path+'/predictions.csv', 'w') as out_file:
    out_file.write(f'%GPR with MC parameter estimation, centred labels, Matern52 kernel\n')
    out_file.write(f'%Pearson_correlation_coefficient:{pearsonsr:.4f},P-value:{pvalue:.4f}\n')
    out_file.write(f'%Spearman_correlation_coefficient:{spearmansr:.4f},P-value:{spvalue:.4f}\n')
    out_file.write(f'%RMSE:{rmse}\n')
    out_file.write(f'%signal_variance:{model.kernel.variance.numpy():.4f},noise_variance:{model.likelihood.variance.numpy():.4f},ARD_lengthscales:{model.kernel.lengthscales.numpy().tolist()}\n')
    out_file.write(f'name,f_pred_mean,f_pred_var,y_true,{",".join(features_test.columns.to_list())}\n')
    for i in range(0, len(mean)):
        out_file.write(f'{features_test.iloc[i].name},{mean[i][0]:.2f},{var[i][0]:.2f},{affinity_test.iloc[i]},{str(features_test.iloc[i].to_list()).strip("[]")}\n')
    out_file.close()


def plot_samples(samples, parameters, y_axis_label):
    plt.figure(figsize=(20, 4))
    for val, param in zip(samples, parameters):
        plt.plot(tf.squeeze(val), label=param_to_name[param])
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("HMC iteration")
    plt.ylabel(y_axis_label)
    plt.savefig(path+'/parameter_evolution_'+y_axis_label+'.png')

plot_samples(samples, model.trainable_parameters, "unconstrained")
plot_samples(parameter_samples, model.trainable_parameters, "constrained")

def marginal_samples(samples, parameters, y_axis_label):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(20, 4), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.savefig(path+'/marginal_parameter_distributions_'+y_axis_label+'.png')

marginal_samples(samples, model.trainable_parameters, "unconstrained")
marginal_samples(parameter_samples, model.trainable_parameters, "constrained")
