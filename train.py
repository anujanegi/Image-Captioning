from data import *
from tokenizer import *
from model import *
from caption import *
from pickle import dump
# training dataset

filename = 'Dataset/Flickr8k_text/Flickr_8k.trainImages.txt'

trainData = loadSet(filename)
trainDescriptions = loadDescriptions('Preprocessed Features/descriptions.txt', trainData)
trainFeatures = loadFeatures('Preprocessed Features/features.pkl', trainData)

tokenizer = tokenizer(trainDescriptions)
dump(tokenizer, open('Preprocessed Features/tokenizer.pkl', 'wb'))
vocabSize = len(tokenizer.word_index) + 1
maxLength = maxLength(trainDescriptions)

# test dataset
#
# filename = 'Dataset/Flickr8k_text/Flickr_8k.devImages.txt'
#
# testData = loadSet(filename)
# testDescriptions = loadDescriptions('Preprocessed Features/descriptions.txt', testData)
# testFeatures = loadFeatures('Preprocessed Features/features.pkl', testData)
#
# X1test, X2test, ytest = createSequence(tokenizer, maxLength, testDescriptions, testFeatures, vocabSize)


def dataGenerator(descriptions, photos, tokenizer, maxLength):
    while 1:
        for key, descriptionList in descriptions.items():
            try:
                photo = photos[key][0]
                inImage, inSequence, outWord = createSequence(tokenizer, maxLength, descriptionList, photo, vocabSize)
                yield [[inImage, inSequence], outWord]
            except:
                pass

model = compileModel(defineModel(vocabSize, maxLength))
epochs = 20
steps = len(trainDescriptions)
for i in range(epochs):
    generator = dataGenerator(trainDescriptions, trainFeatures, tokenizer, maxLength)
    # inputs, outputs = next(generator)
    # print(inputs[0].shape)
    # print(inputs[1].shape)
    # print(outputs.shape)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_'+str(i)+'.h5')
