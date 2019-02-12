import string

# load descriptions file
def loadDescriptionFile(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# extract descriptions and create mapping for each image
def extractDescriptions(file_descriptions):
    descriptions = dict()
    for line in file_descriptions.split('\n'):
        if len(line)<2:
            continue
        imageId, imageDescription = line.split()[0], line.split()[1:]
        imageId = imageId.split('.')[0]
        imageDescription = ' '.join(imageDescription)
        if imageId not in descriptions:
            descriptions[imageId] = list()
        descriptions[imageId].append(imageDescription)

    return descriptions

# clean descriptions
def cleanDescription(descriptions):
    # table for removing descriptions
    table = str.maketrans('', '', string.punctuation)

    for key, descriptionList in descriptions.items():
        for i in range(len(descriptionList)):
            description = descriptionList[i]
            description = description.split()
            description = [word.lower() for word in description]
            description = [i.translate(table) for i in description]
            description = [word for word in description if len(word)>1 and word.isalpha()]
            descriptionList[i] = ' '.join(description)

    return descriptions
