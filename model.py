import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

import numpy as np
import matplotlib.pyplot as plt

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 2

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224

# Common accuracy metric for all outputs, but can use different metrics for different output

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
nb_training = 50

X, Y = np.load('prepared_data/X_folder1.npy'), np.load('prepared_data/Y_folder1.npy')
Y[2] *= 10
# Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4] = Y[:, 0]/224-0.5, Y[:, 1]/224-0.5, Y[:, 2]/11, Y[:, 3]/40-0.5, Y[:, 4]/40-0.5
small_X, small_Y = X[:nb_training], Y[:nb_training]

# for i in range(len(small_X)):
#     plt.imshow(small_X[i])
#     plt.show()
#     print(small_Y[i])

model = Sequential()
# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(5, activation='linear'))
# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

# model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

history = model.fit(X, Y, verbose=1, batch_size=10, epochs=5)
Y_pred = model.predict(X[:1])

plt.plot(history.history['loss'])
plt.show()

model.save('saved_model/poids_folder1.h5')
# print(model.summary())
