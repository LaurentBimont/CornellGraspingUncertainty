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
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
from process_pred import load_my_model, performance, compute_performance
from MCD_uncertainty import make_mse_comparizon, make_classification
from model import model_resnet, resnet_alone
from sklearn.model_selection import train_test_split
import pickle

def process_data():
    # Loading the data
    X_test, Y_test = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
                     np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    # Loading the trained model
    model_name = 'ADAM_bayesian'
    model = model_resnet()
    model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))
    print('Taille du X : ', X_test.shape)
    model_resnet_alone = resnet_alone()
    Y_pred = model.predict(X_test)
    X_traite = model_resnet_alone.predict(X_test)
    good_bad = make_classification(Y_pred, Y_test)
    loss = make_mse_comparizon(Y_pred, Y_test)

    print(len(make_classification(Y_pred, Y_test)))
    print(len(make_mse_comparizon(Y_pred, Y_test)))
    print(compute_performance(model, X_test, Y_test, viz=False))
    print(Y_pred.shape, X_traite.shape)

    result_to_be_compared = []
    for i in range(len(good_bad)):
        result_to_be_compared.append((X_test[i], X_traite[i], Y_test[i], loss[i], good_bad[i]))
    print(len(result_to_be_compared))
    np.save('uncertainty/lossnet_data_aug.npy', np.array(result_to_be_compared))
    return good_bad

########## CONFID NET ##########

def confidnet():
    model = Sequential()
    model.add(Dense(1024, kernel_regularizer=l2(0.01), activation='relu'))
    # model.add(Dropout(rate=0.2))
    # model.add(Dense(1024, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(512, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_confidnet(lossnet_data):
    X, Y = lossnet_data[:, 1], lossnet_data[:, 4]
    X = np.array([np.array(x) for x in X])
    # X = X.reshape((X.shape[0], X.shape[1]))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = confidnet()
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,
                                  patience=50, min_lr=1e-12)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)
    callbacks_list = [reduce_lr, es]
    history = model.fit(X_train, Y_train, batch_size=32, epochs=1000, callbacks=callbacks_list, validation_data=(X_test, Y_test),)
    plt.subplot(1, 2 , 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss evolution')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy evolution')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    model_name = 'ConfidNet_aug_1'
    model_json = model.to_json()
    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
    with open('saved_model/history_{}'.format(model_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


########## LOSS NET ##########
def lossnet():
    model = Sequential()
    model.add(Dense(2048, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.3))
    # model.add(Dense(1024, kernel_regularizer=l2(0.01), activation='relu'))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(512, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='linear'))
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9), loss='mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def train_and_save_lossnet(lossnet_data):
    X, Y = lossnet_data[:, 1], lossnet_data[:, 3]
    X = np.array([np.array(x) for x in X])
    # X = X.reshape((X.shape[0], X.shape[1]))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=42)
    model = lossnet()
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7,
                                  patience=10, min_lr=1e-12)
    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=90)
    callbacks_list = [reduce_lr, es]

    history = model.fit(X_train, Y_train, batch_size=10, epochs=1000, callbacks=callbacks_list, validation_data=(X_test, Y_test))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.show()

    model_name = 'lossNet_3'
    model_json = model.to_json()
    with open("saved_model/model_arch_{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("saved_model/my_model_weights_{}.h5".format(model_name))
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
    # show_history('saved_model/history_ConfidNet_aug.obj')
    # good_bad = process_data()
    # np.save('uncertainty/good_bad_lossnet.npy', good_bad)
    lossnet_data = np.load('uncertainty/lossnet_data_aug.npy', allow_pickle=True)

    train_and_save_lossnet(lossnet_data)
    # train_and_save_confidnet(lossnet_data)
