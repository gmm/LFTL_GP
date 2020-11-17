"""
A script that imports the 2018 PDBbind general set and splits it
into training and testing dataframes. The split is based on which
of the entries belong to the union of the 2007, 2013 and 2016 PDB
core sets and which belong to the 2018 refined set.
The script currently imports 20 features based on a prior feature
relevance analysis and selection.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
top_feature_names = ['BalabanJ', 'BertzCT', 'HallKierAlpha', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
                     'MolLogP', 'NOCount', 'PEOE_VSA1', 'PEOE_VSA3', 'PEOE_VSA6', 'PEOE_VSA7',
                     'PEOE_VSA8', 'SMR_VSA7','as_flex_all', 'cc_A.HD_4', 'cc_N.N_4', 'lig_OA',
                     'vina_gauss2', 'vina_hydrogen', 'vina_repulsion']


def extract_rdkit_features(filepath):
    """
    Extracts the 2018 general set data with most important RDKit features,
    as determined by prior feature relevance analysis.
    Args:
        filepath: path to .csv file with RDKit features

    Returns: pandas dataframe with the desired features
    """

    # this list contains the 20 most important features as ranked in the LFTL paper
    # this was used for analysis before the feature selection was complete
    # lftl_paper_feature_names = ['Unnamed: 0', 'MolMR', 'EState_VSA1', 'MolLogP', 'Chi1v',
    #                            'BertzCT', 'TPSA', 'Chi2n', 'Chi3v', 'ExactMolWt', 'MaxAbsPartialCharge',
    #                            'Chi3n', 'Chi2v', 'PEOE_VSA7', 'MinEStateIndex', 'MinAbsPartialCharge',
    #                            'Chi4n', 'RingCount', 'PEOE_VSA1', 'MaxPartialCharge']

    # the names of the RDKit features contained in the top 20 overall features
    top_feature_names = ['Unnamed: 0', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'MaxAbsEStateIndex',
                         'MaxAbsPartialCharge', 'MolLogP', 'NOCount', 'PEOE_VSA1', 'PEOE_VSA3',
                         'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA7']

    features = pd.read_csv(filepath, index_col=0, usecols=top_feature_names)

    return features


def extract_structural_features(filepath):
    """
    Extracts the 2018 general set data with most important structural features,
    as determined by prior feature relevance analysis.
    Args:
        filepath: path to file with the respective structural features

    Returns: pandas dataframe with structural features
    """

    # this list contains the six Autodock Vina features, which were used for
    # preliminary analysis before the feature selection was complete
    #vina_feature_names = ['Unnamed: 0', 'vina_gauss1', 'vina_gauss2', 'vina_hydrogen',
    #                      'vina_hydrophobic', 'vina_repulsion', 'num_rotors']

    # the names of the structural features out of the top 20 features
    top_structural_feature_names = ['Unnamed: 0', 'as_flex_all', 'cc_A.HD_4',
                                    'cc_N.N_4', 'lig_OA', 'vina_gauss2',
                                    'vina_hydrogen', 'vina_repulsion']

    structural_features = pd.read_csv(filepath, index_col=0, usecols=top_structural_feature_names)

    return structural_features


def extract_binding_affinity(filepath):
    """
    Extracts the Ka values from a given input file
    Args:
        filepath: path to the input file

    Returns: pandas series with binding affinities
    """

    binding_data = pd.read_csv(filepath, index_col=0, squeeze=True)

    return binding_data


def extract_core_set(directory):
    """
    Extracts the indices for the 2007, 2013 and 2016 core sets and
    calculates their union to be used as a validation set.
    Args:
        directory: the directory in which the core sets are stored

    Returns: the union of the PDBbind core sets
    """

    core_sets = {}
    for year in ['2007', '2013', '2016']:
        with open(directory+f'/pdbbind_{year}_core_pdbs.txt') as f:
            core_sets[year] = sorted([l.strip() for l in f])
    core_sets['all'] = [pdb for pdb in core_sets['2007']]
    core_sets['all'] = core_sets['all'] + [pdb for pdb in core_sets['2013'] if pdb not in core_sets['all']]
    core_sets['all'] = core_sets['all'] + [pdb for pdb in core_sets['2016'] if pdb not in core_sets['all']]

    return core_sets


def extract_refined_set(filepath):
    """
    Extracts the indices of the proteins belonging to the refined set.
    Args:
        filename: the location of the file with the indices of the refined set

    Returns: pandas series of the refined set
    """

    refined_set = pd.read_csv(filepath, index_col=0, squeeze=True)

    return refined_set


def extract_data(top_features, sample_size=None, normalize=True, center=True):
    """
    Combines the structural and RDKit features and returns the resulting data set,
    with normalised features/centred labels if desired.
    For different types of analysis, this function allows the user to return only
    the top n (where n<=num_features) features or a randomly sampled subset of
    the training data.
    Args:
        top_features: the number out of the top 20 features to extract, e.g. 5/20
        sample_size: this number of data points to draw out of the whole refined set
        normalize: whether to normalize the features
        center: whether to center the labels

    Returns: two dicts containing the features and labels for the training and testing
             set respectively plus a string column names for referencing specific features

    """

    # read in rdkit and structural features
    rdkit_features = extract_rdkit_features('../data/pdbbind_2018_general_rdkit_features_clean.csv')
    structural_features = extract_structural_features('../data/pdbbind_2018_general_binana_features_clean.csv')

    # join RDKit and structural data
    if rdkit_features.index.equals(structural_features.index):
        features_data = rdkit_features.join(structural_features)
    else:
        raise ValueError("RDKit and Vina training features cannot be joined: incongruent indexing")

    # select the top n features and save the column names
    if top_features > len(top_feature_names) - 1:
        raise ValueError("You requested more top features than are specified in the paper ranking.")

    features_data = features_data[top_features]
    column_order = features_data.columns.values

    # read in binding affinity data
    affinity_data = extract_binding_affinity('../data/pdbbind_2018_general_binding_data_clean.csv')

    # check for which proteins the features and affinity data are available
    # keep only intersection of the two
    if not features_data.index.equals(affinity_data.index):
        intersect = features_data.index.intersection(affinity_data.index)
        features_data = features_data.loc[intersect]
        affinity_data = affinity_data.loc[intersect]
        print("The feature and affinity data have incongruent indexing. Proceeding only with data in intersection.")

    # get the combined refined set for splitting training data
    refined_set = extract_refined_set('refined_set.txt')

    # get the combined core set for splitting testing data
    core_sets = extract_core_set('../data')

    # select elements that are both in the core set and the global set for testing
    test_set = pd.Index(core_sets['all']).intersection(features_data.index)
    # select elements that are in the refined and global set for training,
    # and drop the elements that are already present in the core set
    train_set = refined_set.index.intersection(features_data.index).difference(test_set)

    # split into train and test sets
    features_test = features_data.loc[test_set]
    affinity_test = affinity_data.loc[test_set]

    features_train = features_data.loc[train_set]
    affinity_train = affinity_data.loc[train_set]

    # check for duplicates
    if not features_train.index.intersection(features_test.index).empty:
        raise ValueError('Training and test set are not disjunct.')

    # draw a random sample out of the training data, if specifies
    if sample_size is not None:
        lines = np.random.choice(range(0, len(affinity_train)), size=sample_size)
        features_train = features_train.iloc[lines]
        affinity_train = affinity_train.iloc[lines]

    # normalize both training and testing features, if specified
    # if not, convert them from pd.DataFrame() to np.ndarray() anyway
    if normalize:
        # center and normalise features (doesn't affect prediction quality, but drastically shortens runtime)
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)
    else:
        features_train = features_train.values
        features_test = features_test.values

    # center the training and testing labels if specified
    if center:
        # center labels
        aff_mean = affinity_train.mean()
        affinity_train = affinity_train - aff_mean
        affinity_test = affinity_test - aff_mean

    # create output dictionaries, reshape the training labels to a np.ndarray() (for predictions)
    # but keep the testing labels as a dataframe (to keep indices)
    training_data = {'features': features_train, 'affinity': affinity_train.values.reshape(-1, 1)}
    testing_data = {'features': features_test, 'affinity': affinity_test}

    return training_data, testing_data, column_order
