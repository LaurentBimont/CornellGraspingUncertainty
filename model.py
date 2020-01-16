import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
# Common accuracy metric for all outputs, but can use different metrics for different output

def lr_scheduler(epoch):
    if epoch>100:
        K.set_value(model.optimizer.lr, model.optimizer.lr*0.99)
    return K.get_value(model.optimizer.lr)

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
nb_training = 50

X, Y = np.load('prepared_data/X_train.npy'), np.load('prepared_data/Y_train.npy')

Y[2] *= 10
# Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4] = Y[:, 0]/224-0.5, Y[:, 1]/224-0.5, Y[:, 2]/11, Y[:, 3]/40-0.5, Y[:, 4]/40-0.5
# small_X, small_Y = X[:nb_training], Y[:nb_training]

# for i in range(len(small_X)):
#     plt.imshow(small_X[i])
#     plt.show()
#     print(small_Y[i])
def resnet_model():
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

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model
model = resnet_model()

## Callbacks
lr = LearningRateScheduler(lr_scheduler)
# cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#             filepath="saved_model/poids_ADAM_test.h5",
#             verbose=1
#         )

callbacks_list = [lr]#, cback_checkpoint]

history = model.fit(X, Y, verbose=1, batch_size=32, epochs=1000, callbacks=callbacks_list, shuffle=True)

plt.plot(history.history['loss'])
plt.show()

model_name = 'ADAM_5'
model_json = model.to_json()
with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)

model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
