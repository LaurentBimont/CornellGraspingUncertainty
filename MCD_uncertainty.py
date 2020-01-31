import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error
if __name__=="__main__":
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
import numpy as np
import matplotlib.pyplot as plt
from process_pred import load_my_model, performance
from model import model_resnet


def make_mse_comparizon(Y_pred, Y_test):
    MSE = []
    for x in range(len(Y_pred)):
        diff_mse = []
        for y in Y_test[x]:
            mse = mean_squared_error(np.array(y), Y_pred[x])
            diff_mse.append(mse)
        MSE.append(min(diff_mse))

    return np.array(MSE)


def make_classification(Y_pred, Y_test):
    result_test = []
    for i in range(len(Y_pred)):
        result = False
        for j in range(len(Y_test[i])):
            result = performance(Y_pred[i], Y_test[i][j])
            if result:
                break
        if result:
            result_test.append(1)
        else:
            result_test.append(0)
    return result_test


if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = np.load('prepared_data/X_train.npy', allow_pickle=True), \
                                       np.load('prepared_data/Y_train.npy', allow_pickle=True), \
                                       np.load('prepared_data/X_test.npy', allow_pickle=True), \
                                       np.load('prepared_data/Y_test.npy', allow_pickle=True)

    model_name = 'ADAM_7'
    model = model_resnet(bayesian=True)
    model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))

    # model = load_my_model('saved_model/model_arch_{}.json'.format(model_name), 'saved_model/my_model_weights_{}.h5'.format(model_name))
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    T_bayesian_draw = np.array([f((X_test, 1))[0] for i in range(20)])
    pred = T_bayesian_draw.mean(axis=0)
    mean_var = T_bayesian_draw.var(axis=0).mean(axis=1)
    np.save('uncertainty/mean_var_aug.npy', mean_var)

    result_test = make_classification(pred, Y_test)
    np.save('uncertainty/result_test_aug.npy', result_test)
    MSE = make_mse_comparizon(pred, Y_test)
    np.save('uncertainty/T_bayesian_draw_aug.npy', pred)
    np.save('uncertainty/mse_aug.npy', np.array(MSE))
