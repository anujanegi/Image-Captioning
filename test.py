from pickle import load
from keras.models import load_model
from feature_extraction import *
from caption import *

tokenizer = load(open('Preprocessed Features/tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('model_16.h5')
photo = extractFeature('example.jpg')
desc = description(model, tokenizer, photo, max_length)
print(desc)
