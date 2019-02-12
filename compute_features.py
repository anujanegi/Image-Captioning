from feature_extraction import *
from pickle import dump

# compute features before hand and save in a file
directory = "Dataset/Flicker8k_Dataset"
features = extractFeatures(directory)
dump(features, open('Preprocessed Features/features.pkl', 'wb'))
