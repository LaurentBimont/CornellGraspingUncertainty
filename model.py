import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
if __name__=="__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # tf.enable_eager_execution(config)
    sess = tf.Session(config=config)
    K.set_session(tf.Session(config=config))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

from tensorflow.python.keras.applications import ResNet50, VGG16, DenseNet121, MobileNet, Xception
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization, merge, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from custom_keras import min_mse, neg_log_likelihood, numerateur, denominateur, value_max, value_min
from custom_keras import ConcreteDropout, heteroscedastic_loss


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from time import time

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
nb_training = 50

X, Y = np.load('prepared_data/X_train.npy'), np.load('prepared_data/Y_train.npy')

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

    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=min_mse)
    return model


def model_proper_scoring():
    model = Sequential()
    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(10, activation='linear'))
    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False

    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=neg_log_likelihood, metrics=[numerateur, denominateur, value_max, value_min])
    return model


def ensemble_model(encoder):
    model = Sequential()
    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(encoder(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(5, activation='linear'))
    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss=min_mse)
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=min_mse)
    return model


def train_proper_scoring(model_name):
    model = model_proper_scoring()
    cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="saved_model/checkpoint_{}.h5".format(model_name), save_best_only=True, save_weights_only=True,
        verbose=1, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=5, min_lr=1e-12)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [reduce_lr, es, cback_checkpoint]

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, batch_size=20, epochs=3000,
                        callbacks=callbacks_list, shuffle=True)

    # plt.plot(history.history['loss'])
    # plt.show()

    model_json = model.to_json()
    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)


def concrete_Dropout_model():
    l = 1e-4
    N = 2100              #Nb d'images dans X_train
    wd = l ** 2. / N
    dd = 2. / N
    inputs = Input(shape=(224, 224, 3))
    x = VGG16(include_top=False, pooling='avg', weights='imagenet')(inputs)
    x = ConcreteDropout(Dense(1024, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
    x = ConcreteDropout(Dense(512, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
    mean = ConcreteDropout(Dense(5), weight_regularizer=wd, dropout_regularizer=dd)(x)

    model = Model(inputs, mean)

    model.layers[1].trainable = False

    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss=min_mse)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=min_mse)
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=heteroscedastic_loss)
    return model



def train_ensemble_proper(X_train, X_test, Y_train, Y_test):
    t0 = time()
    T = []
    for j in range(1,5):
        t0 = time()
        train_proper_scoring('proper_scoring_{}'.format(j))
        T.append(time())
    print(T)
    print('Temps total {}'.format(time()-t0))



def train_ensemble(X_train, X_test, Y_train, Y_test):
    model = {"VGG16": VGG16, "Densenet121": DenseNet121, "Resnet50": ResNet50, "MobileNet": MobileNet,
             "Xception": Xception}
    key = 'ProperScore'
    t0 = time()
    T = []
    for j in range(20, 21):
        cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="saved_model/checkpoint_deep_ensemble_{}_{}.h5".format(key, j),
                                                              save_best_only=True, save_weights_only=True, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=5, min_lr=1e-12)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        callbacks_list = [reduce_lr, es, cback_checkpoint]
        t1 = time()
        my_model = ensemble_model(ResNet50)
        history = my_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, batch_size=1, epochs=3000,
                        callbacks=callbacks_list, shuffle=True)
        print('Temps pour entraîner {} est de {}'.format(key, time()-t1))
        T.append('Temps pour entraîner {} est de {}'.format(key, time()-t1))
        model_json = my_model.to_json()

        with open("saved_model/ensemble_{}_{}.json".format(key, j), "w") as json_file:
            json_file.write(model_json)

        with open('saved_model/history_ensemble_{}_{}'.format(key, j), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    print(T)
    print('Temps total {}'.format(time()-t0))


def vgg_alone():
    inputs = Input(shape=(224, 224, 3))
    x = VGG16(include_top=False, pooling='avg', weights='imagenet')(inputs)
    model = Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def resnet_alone():
    inputs = Input(shape=(224, 224, 3))
    x = ResNet50(include_top=False, pooling='avg', weights='imagenet')(inputs)
    model = Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def model_vgg(bayesian=False, dropout=[0.2, 0.2, 0]):
    inputs = Input(shape=(224, 224, 3))
    x = VGG16(include_top=False, pooling='avg', weights='imagenet')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout[0])(x, training=bayesian)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout[1])(x, training=bayesian)
    x = Dense(5, activation='linear')(x)
    x = Dropout(dropout[2])(x, training=bayesian)
    model = Model(inputs, x)
    # model.layers[1].trainable = False
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def model_resnet(bayesian=False):
    inputs = Input(shape=(224, 224, 3))
    x = ResNet50(include_top=False, pooling='avg', weights='imagenet')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x, training=bayesian)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x, training=bayesian)
    x = Dense(5, activation='linear')(x)
    model = Model(inputs, x)
    # model.layers[1].trainable = False
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def model_flipout():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(tfp.layers.DenseFlipout(1024, activation=tf.nn.relu))
    model.add(Dropout(rate=0.2))
    model.add(tfp.layers.DenseFlipout(512, activation=tf.nn.relu))
    model.add(Dropout(rate=0.2))
    model.add(tfp.layers.DenseFlipout(5, activation=tf.nn.relu))
    model.layers[0].trainable = False
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=min_mse)
    return model


def preprocess_data(X, Y):
    Y = np.array([np.array(y[0]) for y in Y])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, Y_train, Y_test


def preprocess_data_min_mse(X, Y):
    Y = np.array([np.array(y) for y in Y])
    Y_max = max([y.shape[0] for y in Y])

    Y_temp = 1000 * np.ones((Y.shape[0], Y_max, 5))
    for i in range(Y_temp.shape[0]):
        Y_temp[i, :Y[i].shape[0], :] = Y[i]
    Y = Y_temp

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, Y_train, Y_test


if __name__=="__main__":
    BATCH_SIZE_TESTING = 1
    nb_training = 50
    model_name = 'ADAM_9_(new_loss)'

    X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
           np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)

    X_train, X_test, Y_train, Y_test = preprocess_data_min_mse(X, Y)

    ## Flipout
    model_name = 'flipout'
    cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="saved_model/checkpoint_{}.h5".format(model_name),
                                                          verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=3, min_lr=1e-12, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    callbacks_list = [reduce_lr, es, cback_checkpoint]

    model = model_flipout()

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, batch_size=20, epochs=3000, shuffle=True,
              callbacks=callbacks_list)

    # model_json = model.to_json()
    # with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
    #     json_file.write(model_json)

    ## Concrete Dropout
    # model_name = 'ConcreteDropout_1'
    # cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath="saved_model/checkpoint_{}.h5".format(model_name),
    #     verbose=1, save_best_only=True, save_weights_only=True
    # )
    #
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
    #                               patience=3, min_lr=1e-12, verbose=1)
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # callbacks_list = [reduce_lr, es, cback_checkpoint]
    #
    # model = concrete_Dropout_model()
    # model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, batch_size=20, epochs=3000,
    #           callbacks=callbacks_list, shuffle=True)

    ## Proper Scoring rule
    # train_proper_scoring(model_name='proper_scoring_1')

    ## Variational Inference
    # model = model_flipout()
    # X_train = X_train.astype(np.float32)
    # pred = model.predict(X_train, batch_size=20)
    # model.fit(X_train, Y_train, batch_size=1)
    # mse = min_mse(tf.convert_to_tensor(Y_train, dtype=np.float32), tf.convert_to_tensor(pred, dtype=np.float32))
    # kl = sum(model.losses)
    # loss = mse + kl
    # train_op = tf.train.AdamOptimizer().minimize(loss)


    ## Deep Ensemble
    # train_ensemble(X_train, X_test, Y_train, Y_test)
    train_ensemble_proper(X_train, X_test, Y_train, Y_test)

    ## Modèle génétation
    # model = resnet_model()
    # model.summary()
    #
    # # cback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # #     filepath="saved_model/checkpoint_{}.h5".format(model_name),
    # #     verbose=1,
    #
    # # )
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
    #                               patience=5, min_lr=1e-12)
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # callbacks_list = [reduce_lr, es]
    #
    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, batch_size=32, epochs=3000,
    #                     callbacks=callbacks_list, shuffle=True)
    #
    # plt.plot(history.history['loss'])
    # plt.show()
    #
    # model_json = model.to_json()
    # with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
    #     json_file.write(model_json)
    #
    # model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
    # with open('saved_model/history_{}'.format(model_name), 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)