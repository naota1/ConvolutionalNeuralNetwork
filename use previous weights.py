# following https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense


# convoluting
classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3),
activation = "relu"))

# pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

# flattening
classifier.add(Flatten())

# create fully connected layer
classifier.add(Dense(units=128, activation="relu"))

# initalize output layer
classifier.add(Dense(units=1, activation="sigmoid"))

# compile model
classifier.compile(optimizer= "adam", loss="binary_crossentropy",
metrics=["accuracy"])

# making predictions
classifier.load("weights.h5")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'