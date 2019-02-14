from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import nltk
from caption import *

def defineModel(vocabSize, maxLength):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(maxLength,))
    se1 = Embedding(vocabSize, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocabSize, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def compileModel(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def evaluateModel(model, descriptions, photos, tokenizer, maxLength):
    actual, predicted = list(), list()
    desc = ''
    for key, descriptionList in descriptions.items():
        try:
            desc = description(model, tokenizer, photos[key], maxLength)
        except:
            pass
        reference = [d.split() for d in descriptionList]
        actual.append(reference)
        predicted.append(desc.split())
    print('BLEU-1: %f' % nltk.translate.bleu_score.sentence_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % nltk.translate.bleu_score.corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % nltk.translate.bleu_score.corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % nltk.translate.bleu_score.corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
