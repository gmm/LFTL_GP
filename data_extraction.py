"""
A script that imports the 2018 PDBbind general set and splits it
into training and testing dataframes. The split is based on which
of the entries belong to the union of the 2007, 2013 and 2016 PDB
core sets and which belong to the 2018 refined.
The script currently imports the six Autodock Vina features and
a given number of the top 20 RDKit features, as listed in the paper
doi: 10.1093/bioinformatics/btz665.
"""


import pandas as pd
rdkit_feature_cutoff = 6


def extract_rdkit_features(filepath, top_n):
    """
    Extracts top n rdkit features (as ranked in the publication) from the stated filepath.
    Args:
        filepath: path to file with RDKit features
        top_n: number of features to extract, starts with the most significant one

    Returns: pandas dataframe with the desired features
    """
    top_feature_names = ['Unnamed: 0', 'MolMR', 'EState_VSA1', 'MolLogP', 'Chi1v',
                         'BertzCT', 'TPSA', 'Chi2n', 'Chi3v', 'ExactMolWt', 'MaxAbsPartialCharge',
                         'Chi3n', 'Chi2v', 'PEOE_VSA7', 'MinEStateIndex', 'MinAbsPartialCharge',
                         'Chi4n', 'RingCount', 'PEOE_VSA1', 'MaxPartialCharge']

    if top_n > len(top_feature_names)-1:
        raise ValueError("You requested more top features than are specified in the paper ranking.")

    features = pd.read_csv(filepath, index_col=0, usecols=top_feature_names[:top_n+1])

    return features


def extract_vina_features(filepath):
    """
    Extract the Autodock Vina features from the specified file path. The rotable bond feature is not
    removed, as it is obviously not present in the top RDKit features.
    Args:
        filepath: path to file with binana features

    Returns: pandas dataframe with Vina features
    """
    vina_feature_names = ['Unnamed: 0', 'vina_gauss1', 'vina_gauss2', 'vina_hydrogen',
                          'vina_hydrophobic', 'vina_repulsion', 'num_rotors']

    vina_features = pd.read_csv(filepath, index_col=0, usecols=vina_feature_names)

    return vina_features


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
    # extract the core set indices
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


# read in rdkit and vina features
rdkit_features = extract_rdkit_features('../data/pdbbind_2018_general_rdkit_features_clean.csv', rdkit_feature_cutoff)
vina_features = extract_vina_features('../data/pdbbind_2018_general_binana_features_clean.csv')

# join RDKit and Vina data
if rdkit_features.index.equals(vina_features.index):
    features_data = rdkit_features.join(vina_features)
else:
    raise ValueError("RDKit and Vina training features cannot be joined: incongruent indexing")

# read in binding affinity data
affinity_data = extract_binding_affinity('../data/pdbbind_2018_general_binding_data_clean.csv')

# check for which proteins the features and affinity data are available
# keep only intersection between the two
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
# select elements that are in the refined and global set, but not already in the core set for training
train_set = refined_set.index.intersection(features_data.index).difference(test_set)


# split into train and test sets
features_test = features_data.loc[test_set]
affinity_test = affinity_data.loc[test_set]

features_train = features_data.loc[train_set]
affinity_train = affinity_data.loc[train_set]
