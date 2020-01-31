import tensorflow as tf
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
from model import resnet_model


def load_my_model(arch_file, weight_file):
    # with open(arch_file, 'r') as json_file:
    #     architecture = json.load(json_file)
    #     model = model_from_json(json.dumps(architecture))
    # model = model_from_json(arch_file)
    model = resnet_model()
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
    '''
    :param grasp_pred: [[[x0,y0],[x0,y0]],...[]]
    :param grasp_true:
    :return:
    '''
    grasp_pred = grasp_to_bbox(Y_pred)
    grasp_true = grasp_to_bbox(Y_true)

    p_pred = Polygon(grasp_pred)
    p_true = Polygon(grasp_true)

    iou = p_pred.intersection(p_true).area / (p_pred.area + p_true.area - p_pred.intersection(p_true).area)

    # tan_pred, tan_true = np.arctan(Y_pred[2])*180/np.pi, np.arctan(Y_true[2])*180/np.pi
    # print(iou, np.abs(tan_pred-tan_true))
    # if iou > 0.25 and np.abs(tan_pred-tan_true) < 30:
    #     print(iou, np.abs(tan_pred - tan_true), True)
    theta_pred, theta_true = Y_pred[2], Y_true[2]
    if iou > 0.25 and np.abs(theta_pred-theta_true) < 30:
        # print(iou, np.abs(theta_pred-theta_true), True)
        return True
    else:
        # print(iou, np.abs(theta_pred-theta_true), False)
        return False

def compute_performance(model, X_test, Y_test, viz=False):
    Y_pred = model.predict(X_test)
    score = 0
    for i in range(len(X_test)):
        for j in range(len(Y_test[i])):
            result = performance(Y_pred[i], Y_test[i][j])
            if viz:
                vizualize_pred(X_test[i], Y_test[i][j], Y_pred[i])
            if result:
                score += 1
                break
    print('Il y a eu {} de grasp réussis sur {}.'.format(score, len(X_test)))
    return score/len(X_test)

if __name__=="__main__":
    X_train, Y_train, X_test, Y_test = np.load('prepared_data/X_train.npy', allow_pickle=True),\
                                       np.load('prepared_data/Y_train.npy', allow_pickle=True),\
                                       np.load('prepared_data/X_test.npy', allow_pickle=True),\
                                       np.load('prepared_data/Y_test.npy', allow_pickle=True)
    # grasp_pred, grasp_true, Y_pred = np.load('pred_rectangle.npy'),np.load('true_rectangle.npy'), np.load('y_pred.npy')
    X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True), \
           np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model_name = 'ADAM_7'
    model = load_my_model('saved_model/model_arch_{}.json'.format(model_name), 'saved_model/my_model_weights_{}.h5'.format(model_name))
    print(compute_performance(model, X_test, Y_test, viz=False))
