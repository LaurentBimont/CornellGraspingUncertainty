import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from process_data import grasp_to_bbox, draw_rectangle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

X, Y = np.load('prepared_data/X_folder1.npy'), np.load('prepared_data/Y_folder1.npy')

def load_predict(X, Y, viz=False):
    model = load_model('saved_model/poids_folder1.h5')
    y_pred = model.predict(X[:10])
    y_pred[:, 2] /= 10
    np.save('y_pred.npy', y_pred)
    if viz:
        for i in range(10):
            vizualize_pred(X[i], Y[i], y_pred[i])
    return y_pred

def vizualize_pred(image_entree, y_true, y_pred):
    grasp_pred = grasp_to_bbox(y_pred)
    grasp_true = grasp_to_bbox(y_true)
    print('y_pred', y_pred)
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
    plt.scatter(y_true[0], y_pred[0][1])
    plt.imshow(image_true)
    plt.show()

    np.save('pred_rectangle.npy', grasp_pred)
    np.save('true_rectangle.npy', grasp_true)

# vizualize_pred()
grasp_pred, grasp_true, Y_pred = np.load('pred_rectangle.npy'),np.load('true_rectangle.npy'), np.load('y_pred.npy')

def performance(Y_pred, Y_true):
    '''
    :param grasp_pred: [[[x0,y0],[x0,y0]],...[]]
    :param grasp_true:
    :return:
    '''

    grasp_pred = grasp_to_bbox(*Y_pred)
    grasp_true = grasp_to_bbox(Y_true)

    p_pred = Polygon(grasp_pred)
    p_true = Polygon(grasp_true)

    iou = p_pred.intersection(p_true).area / (p_pred.area + p_true.area - p_pred.intersection(p_true).area)

    tan_pred, tan_true = np.arctan(Y_pred[0][2])*180/np.pi, np.arctan(Y_true[2])*180/np.pi
    print(iou, tan_pred, tan_true)
    if iou > 0.25 and np.abs(tan_pred-tan_true) < 30:
        return True
    else:
        return False


load_predict(X, Y, viz=True)
# vizualize_pred()
performance(Y_pred, Y[0])
