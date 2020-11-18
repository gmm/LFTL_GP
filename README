# Gaussian Process regression on the PDBbind datasets

Boyles et. al. (Bioinformatics, 36(3), 2020, 758–76) showed that 
extending protein-ligand based interactional features with purely 
ligand based features improves 
the performance of machine learning algorithms for universal affinity 
prediction. We extended their evaluation of random forest regressors 
to Gaussian processes, which allow for the native modelling of predictive 
uncertainty.

## Requirements

Running the code requires the GPflow package (https://github.com/GPflow/GPflow) and
the work of Boyles et. al. available here (http://opig.stats.ox.ac.uk/resources).
No other libraries than the ones required to run these two are required.

## Using the code

The code trains Gaussian Process regressors using the 20 most relevant 
features, as determined by a prior feature relevance analysis of the features identified 
in Boyles et. al..

To run a hyperparameter grid search in order to find optimal starting 
parameters, run the respective function in hyperparameter_grid_search.py.
To examine the effect of an increasing feature number or training set size, run 
the respective functions in feature_data_exploration. A short demo is given in
main.py.

All relevant plotting functions are in utilities.py.
