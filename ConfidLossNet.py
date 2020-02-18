from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.keras.backend as K
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    K.set_session(tf.Session(config=config))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.enable_eager_execution(config)
from tensorflow.python.keras.applications import ResNet50, VGG16
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
from process_pred import load_my_model, performance, compute_performance
from MCD_uncertainty import make_mse_comparizon, make_classification
from model import model_resnet, resnet_alone, ensemble_model, vgg_alone
from sklearn.model_selection import train_test_split
import pickle
from time import time
from model import concrete_Dropout_model


def process_data(model, pre_model, model_name):
    # Loading the data
    # X_test, Y_test = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
    #                  np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
           np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # Loading the trained model
    # model = model_resnet()
    # model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))
    # print('Taille du X : ', X.shape)
    # model_resnet_alone = resnet_alone()
    Y_pred = model.predict(X, batch_size=20)
    X_traite = pre_model.predict(X, batch_size=20)
    good_bad = make_classification(Y_pred, Y)
    loss = make_mse_comparizon(Y_pred, Y)

    print(len(make_classification(Y_pred, Y)))
    print(len(make_mse_comparizon(Y_pred, Y)))
    print('Performance sur les données de test : ', compute_performance(model, X_test, Y_test, viz=False))
    print(Y_pred.shape, X_traite.shape)

    result_to_be_compared = []
    for i in range(len(good_bad)):
        result_to_be_compared.append((X[i], X_traite[i], Y[i], loss[i], good_bad[i]))
    np.save('uncertainty/lossnet_data_aug_{}.npy'.format(model_name), np.array(result_to_be_compared))
    return good_bad


def process_data_proper(pre_model, model_name):
    # Loading the data
    # X_test, Y_test = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
    #                  np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
           np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    Y_tot = []
    for i in range(0, 5):
        model = load_my_model('saved_model/model_arch_proper_scoring_{}.json'.format(i),
                              'saved_model/checkpoint_proper_scoring_{}.h5'.format(i))
        Y_tot.append(model.predict(X_test, batch_size=20))
        del model
    Y_tot = np.array(Y_tot)
    Y_pred = Y_tot.mean(axis=0)[:, :5]


    X_traite = pre_model.predict(X, batch_size=20)
    good_bad = make_classification(Y_pred, Y)
    loss = make_mse_comparizon(Y_pred, Y)

    print(len(make_classification(Y_pred, Y)))
    print(len(make_mse_comparizon(Y_pred, Y)))
    # print('Performance sur les données de test : ', compute_performance(model, X_test, Y_test, viz=False))
    print(Y_pred.shape, X_traite.shape)

    result_to_be_compared = []
    for i in range(len(good_bad)):
        result_to_be_compared.append((X[i], X_traite[i], Y[i], loss[i], good_bad[i], Y_tot[i]))
    np.save('uncertainty/lossnet_data_aug_{}.npy'.format(model_name), np.array(result_to_be_compared))
    return good_bad



########## CONFID NET ##########
def weighted_binary_crossentropy(y_true, y_pred):
    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    # Calculate the binary crossentropy
    # nb_bad, nb_good, nb_tot = K.size(y_pred) - K.sum(y_pred), K.sum(y_pred), K.size()
    # one_weight, zero_weight = nb_bad/nb_tot, nb_good/nb_tot
    # one_weight, zero_weight = 149/2947, 1-(149/2947)
    one_weight, zero_weight = 1, 12
    b_ce = K.binary_crossentropy(y_true, y_pred)
    # Apply the weights
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    # Return the mean error
    return K.mean(weighted_b_ce)

def confidnet(Restrainable=False):
    model = Sequential()
    if Restrainable:
        model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(2048, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1024, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(512, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss=weighted_binary_crossentropy, metrics=['accuracy'])
    return model

def train_and_save_confidnet(lossnet_data, model_name, Restrainable):
    if not Restrainable:
        X, Y = lossnet_data[:, 1], lossnet_data[:, 4]
    else:
        X, Y = lossnet_data[:, 0], lossnet_data[:, 4]
    X = np.array([np.array(x) for x in X])

    nb_good, nb_bad = sum(Y), len(Y)-sum(Y)
    print(nb_good, nb_bad)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = confidnet(Restrainable=Restrainable)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.9,
                                  patience=5, min_lr=1e-12)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
    mc = ModelCheckpoint('saved_model/checkpoint_{}.h5'.format(model_name), monitor='val_loss', verbose=1,
                         save_best_only=True,
                         save_weights_only=True, mode='auto', period=1)
    callbacks_list = [reduce_lr, es, mc]
    t0 = time()
    history = model.fit(X_train, Y_train, batch_size=20, epochs=100, callbacks=callbacks_list, validation_data=(X_test,
                                                                                                                 Y_test))
    print('Temps d\'entraînement : {}'.format(time() - t0))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss evolution')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('Accuracy evolution')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], label='learning rate evolution')
    plt.show()
    model_json = model.to_json()

    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)

    # model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
    with open('saved_model/history_{}'.format(model_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


########## LOSS NET ##########
def lossnet(Restrainable=True):
    model = Sequential()
    if Restrainable:
        model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1024, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.2))  #, kernel_regularizer=l2(0.01)
    model.add(Dense(1, activation='linear'))
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model

def train_and_save_lossnet(lossnet_data, model_name, Restrainable):
    if not Restrainable:
        X, Y = lossnet_data[:, 1], lossnet_data[:, 3]
    else:
        X, Y = lossnet_data[:, 0], lossnet_data[:, 3]
    X = np.array([np.array(x) for x in X])
    # X = X.reshape((X.shape[0], X.shape[1]))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = lossnet(Restrainable=Restrainable)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=5, min_lr=1e-12)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    mc = ModelCheckpoint('saved_model/checkpoint_{}.h5'.format(model_name), monitor='val_loss', verbose=1, save_best_only=True,
                                              save_weights_only=True, mode='auto', period=1)
    callbacks_list = [reduce_lr, es, mc]

    t0 = time()

    model_json = model.to_json()
    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)

    history = model.fit(X_train, Y_train, batch_size=20, epochs=1000, callbacks=callbacks_list, validation_data=(X_test, Y_test))
    print('Temps d\'entraînement : {}'.format(time()-t0))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.show()

    model_json = model.to_json()
    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    # model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
    with open('saved_model/history_{}'.format(model_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


####### Show history
def show_history(path):
    f = open(path, 'rb')
    history = pickle.load(f)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.suptitle('Training evolution')
    plt.title('Accuracy ')
    plt.plot(history['acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='test accuracy')
    plt.ylim([0.5, 1])
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('Loss ')
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='test loss')
    plt.legend()
    plt.ylim([0., 1])
    plt.subplot(1, 3, 3)
    plt.title('Learning rate')
    plt.plot(history['lr'], label='Learning rate evolution')
    plt.legend()
    plt.show()

if __name__=='__main__':
    pre_model = vgg_alone()
    process_data_proper(pre_model=pre_model, model_name='proper_ensemble')


    # show_history('saved_model/history_ConfidNet_aug.obj')


    # model = concrete_Dropout_model()
    # model.load_weights('saved_model/checkpoint_ConcreteDropout_1.h5')
    #
    # # model = load_my_model('saved_model/ensemble_VGG16_6.json', 'saved_model/checkpoint_deep_ensemble_VGG16_6.h5')
    # pre_model = vgg_alone()
    # good_bad = process_data(model, pre_model, 'VGG16_concrete_dropout')
    # np.save('uncertainty/lossnet_vgg_concrete_dropout.npy', good_bad)
    # # np.save('uncertainty/good_bad_lossnet.npy', good_bad)
    # # process_data(model_name='ADAM_8')
    # lossnet_data = np.load('uncertainty/lossnet_data_aug_VGG16_6.npy', allow_pickle=True)
    #
    # train_and_save_lossnet(lossnet_data, model_name='LossNet_10', Restrainable=True)
    # train_and_save_confidnet(lossnet_data, model_name='ConfidNet_10', Restrainable=True)
    # show_history('saved_model/history_ConfidNet_5')
