from pickle import load

def loadFile(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data

def loadSet(filename):
    file = loadFile(filename)
    data = list()
    for line in file.split('\n'):
    	if len(line) < 1:
    		continue
    	identifier = line.split('.')[0]
    	data.append(identifier)

    return set(data)

def loadDescriptions(filename, dataset):
    data = loadFile(filename)
    descriptions = dict()
    for line in data.split('\n'):
    	imageId, imageDescription = line.split()[0], line.split()[1:]
    	if imageId in data:
    		if imageId not in descriptions:
    			descriptions[imageId] = list()
    		description = 'startseq ' + ' '.join(imageDescription) + ' endseq'
    		descriptions[imageId].append(description)

    return descriptions

def loadFeatures(filename, data):
    features = load(open(filename, 'rb'))
    features = {i: features[i] for i in data}

    return features
