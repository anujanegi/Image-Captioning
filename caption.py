from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

def wordById(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None

def description(model, tokenizer, photo, maxLength):
    inText = 'startseq'
    for i in range(maxLength):
        sequence = tokenizer.texts_to_sequences([inText])[0]
        sequence = pad_sequences([sequence], maxlen = maxLength)
        nextWord = model.predict([photo, sequence], verbose=0)
        nextWord = argmax(nextWord)
        word = wordById(nextWord, tokenizer)
        if word is None:
            break
        inText += ' ' + word
        if word == 'endseq':
            break

    return inText
