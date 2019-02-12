from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def extractFeatures(directory):

    # defining model
    model = VGG16()
    # removing the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())

    # feature extraction from photos
    features = dict()
    for file in listdir(directory):
        filename = directory + '/' + file
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        features[file.split('.')[0]] = model.predict(image, verbose = 0)

    return features
