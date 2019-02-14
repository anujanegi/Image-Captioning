from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

def toLines(descriptions):
    description = list()
    for key in descriptions.keys():
        [description.append(d) for d in descriptions[key]]

    return description

def tokenizer(descriptions):
    description = toLines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(description)

    return tokenizer

def createSequence(tokenizer, maxLength, descriptionList, photos, vocabSize):
    X1, X2, y = list(), list(), list()
    for description in descriptionList:
        sequence = tokenizer.texts_to_sequences([description])[0]
        for i in range(1, len(sequence)):
            inSequence, outSequence = sequence[:i], sequence[i]
            inSequence = pad_sequences([inSequence], maxlen=maxLength)[0]
            outSequence = to_categorical([outSequence], num_classes=vocabSize)[0]
            X1.append(photos)
            X2.append(inSequence)
            y.append(outSequence)

    return array(X1), array(X2), array(y)

def maxLength(descriptions):
    lines = toLines(descriptions)

    return max(len(d.split()) for d in lines)
