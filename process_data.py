import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import os
import cv2
print(os.getcwd())

def process(points):
    '''
    :param points: Lignes du fichier
    :return: Liste de listes contenant les 4 coins de chaque rectangle
    '''
    points = [point.rstrip(' \n').split(' ') for point in points]
    rectangle_vertices = []
    for i in range(len(points)//4):
        rectangle_vertices.append(points[4*i:4*(i+1)])
    return rectangle_vertices

def draw_rectangle(mes_rectangles, image):
    for rectangle in mes_rectangles:
        print(rectangle)
        point1, point2 = tuple([int(float(point)) for point in rectangle[0]]), tuple(
            [int(float(point)) for point in rectangle[1]])
        point3, point4 = tuple([int(float(point)) for point in rectangle[2]]), tuple(
            [int(float(point)) for point in rectangle[3]])
        cv2.line(image, point1, point2, color=(255, 0, 0), thickness=3)
        cv2.line(image, point2, point3, color=(0, 255, 0), thickness=3)
        cv2.line(image, point3, point4, color=(255, 0, 0), thickness=3)
        cv2.line(image, point4, point1, color=(0, 255, 0), thickness=3)
    return image

def vizualise(folder):
    for i in range(100, 200):
        image = imread(folder+'/pcd0{}r.png'.format(i))
        image_good, image_bad = np.copy(image), np.copy(image)

        f = open(folder+'/pcd0{}cpos.txt'.format(i))
        points = f.readlines()
        rectangles = process(points)
        image_good = draw_rectangle(rectangles, image_good)
        plt.subplot(1, 2, 1)
        plt.imshow(image_good)
        plt.title('Good grasping rectangles')

        f = open(folder+'/pcd0{}cneg.txt'.format(i))
        points = f.readlines()
        rectangles = process(points)
        image_bad = draw_rectangle(rectangles, image_bad)
        plt.subplot(1, 2, 2)
        plt.imshow(image_bad)
        plt.title('Bad grasping rectangles')
        plt.show()


def zoom_on_data(viz=False):
    j = 0
    start_x, end_x, start_y, end_y = 124, 418, 104, 398
    for k in range(1, 2):
        for i in range(100, 200):
            image = imread('folder{}/pcd0{}r.png'.format(k, i))
            image = image[start_x:end_x, start_y:end_y, :]
            f = open('folder{}/pcd0{}cpos.txt'.format(k, i), 'r')
            points = f.readlines()

            old_shape = image.shape
            image = cv2.resize(image, (224, 224))
            new_shape = image.shape
            x_ratio, y_ratio = new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]
            f.close()
            points = [str(x_ratio*(float(point.split(' ')[0])-start_y))+' '+str(x_ratio*(float(point.split(' ')[1])-start_x))+'\n' for point in points]
            if viz:
                rectangles = process(points)
                image_good = draw_rectangle(rectangles, image)
                plt.imshow(image_good)
                plt.title('Good grasping rectangles')
                plt.show()
            f = open('processed_data/pcd0{}cpos.txt'.format(j), 'w')
            f.writelines(points)
            imsave('processed_data/pcd0{}r.png'.format(j), image)
            j += 1


def bboxes_to_grasps(box):
    # bboxes au format [x0, y0, x1, y1, x2, y2, x3, y3]
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    x = (box[0] + (box[4] - box[0])/2)
    y = (box[1] + (box[5] - box[1])/2)

    if box[0] == box[2]:
        tan = 30
    else:
        tan = (box[3] - box[1]) / (box[2] - box[0])
    tan = max(-11, min(tan, 11))
    h = np.sqrt(np.power((box[2] - box[0]), 2) + np.power((box[3] - box[1]), 2))
    w = np.sqrt(np.power((box[6] - box[0]), 2) + np.power((box[7] - box[1]), 2))
    return x, y, tan, h, w


def grasp_to_bbox(grasp):
    x, y, tan, h, w = grasp
    theta = np.arctan(tan)
    edge1 = [x - w/2*np.cos(theta) + h/2*np.sin(theta), y - w/2*np.sin(theta) - h/2*np.cos(theta)]
    edge2 = [x + w/2*np.cos(theta) + h/2*np.sin(theta), y + w/2*np.sin(theta) - h/2*np.cos(theta)]
    edge3 = [x + w/2*np.cos(theta) - h/2*np.sin(theta), y + w/2*np.sin(theta) + h/2*np.cos(theta)]
    edge4 = [x - w/2*np.cos(theta) - h/2*np.sin(theta), y - w/2*np.sin(theta) + h/2*np.cos(theta)]
    return [edge1, edge2, edge3, edge4]


def create_preprocessing_data(path, viz=False):
    X, Y = [], []
    A = [file for file in os.listdir(path) if ('.png' in file)]
    for image_name in A:
        image = imread(path+'/'+image_name)
        number = image_name.rstrip('r.png')
        f = open(path + '/' + number + 'cpos.txt')
        grasp_points = f.readlines()
        for nb_grasp in range(len(grasp_points)//4):
            box = []
            for i in range(4):
                temp_grasp = list(map(float, grasp_points[nb_grasp*4+i].split(" ")))
                box.append(temp_grasp[0])
                box.append(temp_grasp[1])
            grasp_param = bboxes_to_grasps(box)

            print(grasp_param, [grasp==grasp for grasp in grasp_param])
            if viz:
                plt.imshow(image)
                plt.scatter(grasp_param[0], grasp_param[1])
                plt.show()
            nan_check = [grasp == grasp for grasp in grasp_param]
            if False not in nan_check:
                X.append(image)
                Y.append(grasp_param)

    return np.array(X), np.array(Y)


if __name__=="__main__":
    zoom_on_data(viz=False)

    X, Y = create_preprocessing_data('processed_data')
    np.save('prepared_data/X_folder1.npy', X)
    np.save('prepared_data/Y_folder1.npy', Y)
