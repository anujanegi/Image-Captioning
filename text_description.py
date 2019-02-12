from preprocessing_descriptions import *

# get descriptions
filename = "Dataset/Flickr8k_text/Flickr8k.token.txt"
descriptions = cleanDescription(extractDescriptions(loadDescriptionFile(filename)))

# save descriptions
filename = 'Preprocessed Features/descriptions.txt'
data = list()
for key, descriptionList in descriptions.items():
    for description in descriptionList:
        data.append(key + ' ' + description)
data = '\n'.join(data)
file = open(filename, 'w')
file.write(data)
file.close()
