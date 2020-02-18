import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import set_session
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    # config = tf.ConfigProto()
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # config.gpu_options.allow_growth = True
    # tf.enable_eager_execution(config)

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import numpy as np
from process_data import grasp_to_bbox, draw_rectangle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import json
import tqdm
from model import resnet_model, concrete_Dropout_model, model_flipout


def load_my_model(arch_file=None, weight_file=None):
    if arch_file is None:
        model = resnet_model()
    with open(arch_file, 'r') as json_file:
        architecture = json.load(json_file)
        model = model_from_json(json.dumps(architecture))
        # model = model_from_json(arch_file)

    model.load_weights(weight_file)
    return model

def vizualize_pred(image_entree, y_true, y_pred):
    grasp_pred = grasp_to_bbox(y_pred)
    grasp_true = grasp_to_bbox(y_true)
    image = np.copy(image_entree)
    image_pred = draw_rectangle([grasp_pred], image)

    plt.subplot(1, 2, 1)
    plt.imshow(image_pred)
    plt.scatter(y_pred[0], y_pred[1])
    plt.title('Prediction')

    image = np.copy(image_entree)
    image_true = draw_rectangle([grasp_true], image)
    plt.subplot(1, 2, 2)
    plt.title('True')
    plt.scatter(y_true[0], y_true[1])
    plt.imshow(image_true)
    plt.show()

    # np.save('pred_rectangle.npy', grasp_pred)
    # np.save('true_rectangle.npy', grasp_true)


def performance(Y_pred, Y_true):
    grasp_pred = grasp_to_bbox(Y_pred)
    grasp_true = grasp_to_bbox(Y_true)

    p_pred = Polygon(grasp_pred)
    p_true = Polygon(grasp_true)

    iou = p_pred.intersection(p_true).area / (p_pred.area + p_true.area - p_pred.intersection(p_true).area)
    theta_pred, theta_true = Y_pred[2], Y_true[2]
    if iou > 0.25 and (np.abs(theta_pred-theta_true) < 30 or np.abs(theta_pred % 180-theta_true % 180)):
        return True
    else:
        return False


def compute_performance_ensemble_proper(X_test, Y_test, viz=False):
    Y_tot = []
    for i in range(0, 5):
        model = load_my_model('saved_model/model_arch_proper_scoring_{}.json'.format(i),
                              'saved_model/checkpoint_proper_scoring_{}.h5'.format(i))
        Y_tot.append(model.predict(X_test, batch_size=20))
        del model
    Y_tot = np.array(Y_tot)
    Y_pred = Y_tot.mean(axis=0)[:, :5]
    score, res = 0, []

    for i in range(len(X_test)):
        goodornot = False
        for j in range(len(Y_test[i])):
            result = performance(Y_pred[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_pred[i])
            if result:
                goodornot = True
                score += 1
                break
        if goodornot:
            res.append(1)
        else:
            res.append(0)
    np.save('uncertainty/deep_ensemble_proper_output_5_VGG16.npy', Y_tot)
    np.save('uncertainty/deep_ensemble_proper_result_5_VGG16.npy', res)
    print('Il y a eu {} de grasp réussis sur {} équivalent à {}'.format(score, len(X_test), score / len(X_test)))
    return score / len(X_test)


def compute_performance_ensemble(X_test, Y_test, viz=False):
    model_ensemble = ["VGG16", "Densenet121", "Resnet50", "MobileNet", "Xception"]
    Y_tot = []
    # for member in model_ensemble:
    #     model = load_my_model('saved_model/ensemble_{}_1.json'.format(member),
    #                           'saved_model/checkpoint_deep_ensemble_{}_2.h5'.format(member))
    #     compute_performance(model, X_test, Y_test)
    #     Y_tot.append(model.predict(X_test, batch_size=20))
    #     del model
    for i in range(2, 7):
        model = load_my_model('saved_model/ensemble_VGG16_1.json'.format(i),
                               'saved_model/checkpoint_deep_ensemble_VGG16_{}.h5'.format(i))
        compute_performance(model, X_test, Y_test)
        Y_tot.append(model.predict(X_test, batch_size=20))
        del model
    Y_tot = np.array(Y_tot)
    Y_tot = Y_tot.mean(axis=0)

    score, res = 0, []
    for i in range(len(X_test)):
        goodornot = False
        for j in range(len(Y_test[i])):
            result = performance(Y_tot[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_tot[i])
            if result:
                goodornot = True
                score += 1
                break
        if goodornot:
            res.append(1)
        else:
            res.append(0)
    np.save('uncertainty/deep_ensemble_output_5_VGG16.npy', Y_tot)
    np.save('uncertainty/deep_ensemble_result_5_VGG16.npy', res)
    print('Il y a eu {} de grasp réussis sur {}.'.format(score, len(X_test)))
    return score/len(X_test)


def compute_performance_flipout(X_test, Y_test, viz=False):
    Y_tot = []
    model = model_flipout()
    model.load_weights('saved_model/checkpoint_flipout.h5')
    for i in tqdm.tqdm(range(100)):
        Y_tot.append(model.predict(X_test, batch_size=20))
    Y_tot = np.array(Y_tot)
    Y_tot = Y_tot.mean(axis=0)

    score, res = 0, []
    for i in tqdm.tqdm(range(len(X_test))):
        goodornot = False
        for j in range(len(Y_test[i])):
            result = performance(Y_tot[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_tot[i])
            if result:
                goodornot = True
                score += 1
                break
        if goodornot:
            res.append(1)
        else:
            res.append(0)
    np.save('uncertainty/flipout_output_5_VGG16.npy', Y_tot)
    np.save('uncertainty/flipout_result_5_VGG16.npy', res)
    print('Il y a eu {} de grasp réussis sur {}.'.format(score, len(X_test)))
    return score/len(X_test)


def compute_performance_proper(model, X_test, Y_test, viz=False):
    Y_pred_tot = model.predict(X_test, batch_size=20)
    Y_pred = Y_pred_tot[:, :5]
    score, res = 0, []
    for i in range(len(X_test)):
        goodornot = False
        for j in range(len(Y_test[i])):
            result = performance(Y_pred[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_pred[i])

            if result:
                goodornot = True
                score += 1
                break
        if goodornot:
            res.append(1)
        else:
            res.append(0)
    np.save('uncertainty/proper_score_output_1.npy', Y_pred_tot)
    np.save('uncertainty/proper_score_result_1.npy', res)
    print('Il y a eu {} grasp réussis sur {} équivalent à {}'.format(score, len(X_test), score / len(X_test)))
    return score / len(X_test)


def compute_performance(model, X_test, Y_test, viz=False):
    Y_pred = model.predict(X_test, batch_size=20)
    score = 0
    for i in range(len(X_test)):
        for j in range(len(Y_test[i])):
            result = performance(Y_pred[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_pred[i])
            if result:
                score += 1
                break
    print('Il y a eu {} de grasp réussis sur {} équivalent à {}'.format(score, len(X_test), score/len(X_test)))
    return score/len(X_test)

def concreteDropout_eval(model):
    ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
    print(ps)


if __name__=="__main__":
    X_train, Y_train, X_test, Y_test = np.load('prepared_data/X_train.npy', allow_pickle=True),\
                                       np.load('prepared_data/Y_train.npy', allow_pickle=True),\
                                       np.load('prepared_data/X_test.npy', allow_pickle=True),\
                                       np.load('prepared_data/Y_test.npy', allow_pickle=True)

    # grasp_pred, grasp_true, Y_pred = np.load('pred_rectangle.npy'),np.load('true_rectangle.npy'), np.load('y_pred.npy')
    X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True), \
           np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # model = load_my_model('saved_model/ensemble_ResNet50_10.json'.format(model_name),
    #                       'saved_model/checkpoint_deep_ensemble_ResNet50_17.h5'.format(model_name))

    ### Flipout
    compute_performance_flipout(X_test, Y_test)

    ### Ensemble + Proper
    # compute_performance_ensemble_proper(X_test, Y_test, viz=False)

    ### Concrete Dropout
    # model = concrete_Dropout_model()
    # model.load_weights('saved_model/checkpoint_ConcreteDropout_1.h5')
    # f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # model.predict(X_test[:1])
    # ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
    # print(ps)
    # print(compute_performance(model, X_test, Y_test, viz=False))
    # print(compute_performance_ensemble(X_test, Y_test, viz=False))
    # print(compute_performance_proper(model, X_test, Y_test, viz=False))
