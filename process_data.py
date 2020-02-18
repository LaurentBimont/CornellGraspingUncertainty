import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import os
import cv2
import sys
import tqdm


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
        point1, point2 = tuple([int(float(point)) for point in rectangle[0]]), tuple(
            [int(float(point)) for point in rectangle[1]])
        point3, point4 = tuple([int(float(point)) for point in rectangle[2]]), tuple(
            [int(float(point)) for point in rectangle[3]])
        cv2.line(image, point1, point2, color=(0, 255, 0), thickness=3)
        cv2.line(image, point2, point3, color=(255, 0, 0), thickness=3)
        cv2.line(image, point3, point4, color=(0, 255, 0), thickness=3)
        cv2.line(image, point4, point1, color=(255, 0, 0), thickness=3)
        cv2.circle(image, point1, radius=5, color=(0, 0, 255))
        cv2.circle(image, point2, radius=1, color=(0, 0, 255))
    return image


def vizualise(folder):
    for i in range(00, 950):
        image = imread(folder+'/pcd0{}r.png'.format(i))
        image_good, image_bad = np.copy(image), np.copy(image)
        f = open(folder+'/pcd0{}cpos.txt'.format(i))
        points = f.readlines()
        rectangles = process(points)
        image_good = draw_rectangle(rectangles, image_good)
        plt.subplot(1, 2, 1)
        plt.imshow(image_good)
        plt.title('Good grasping rectangles')
        NewRect = []
        for rect in rectangles:
            rect = [float(item) for vertex in rect for item in vertex]
            grasp = bboxes_to_grasps(rect)
            new_rect = grasp_to_bbox(grasp)
            NewRect.append(new_rect)

        image_bad = draw_rectangle(NewRect, image_bad)
        plt.subplot(1, 2, 2)
        plt.imshow(image_bad)
        plt.title('Double transformed grasping rectangles')
        # f = open(folder+'/pcd0{}cneg.txt'.format(i))
        # points = f.readlines()
        # rectangles = process(points)
        # image_bad = draw_rectangle(rectangles, image_bad)
        # plt.subplot(1, 2, 2)
        # plt.imshow(image_bad)
        # plt.title('Bad grasping rectangles')
        plt.show()


def vizualise_all(X, Y):
    for x, list_y in zip(X, Y):
        for y in list_y:
            draw_rectangle([grasp_to_bbox(y)], x)
        plt.imshow(x)
        plt.show()



def zoom_on_data(viz=False):
    j = 0
    start_x, end_x, start_y, end_y = 124, 418, 144, 438
    for k in range(1, 2):
        for i in range(k*100, (k+1)*100):
            try:
                image = imread('folder0{}/pcd0{}r.png'.format(k, i))
                image = image[start_x:end_x, start_y:end_y, :]
                f = open('folder0{}/pcd0{}cpos.txt'.format(k, i), 'r')
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
            except:
                pass


def vizualize_jacquard():
    j = 0
    for i in tqdm.tqdm(range(12)):
        print(i)
        for dir_name in tqdm.tqdm(os.listdir('Jacquard/Jacquard_Dataset_{}'.format(i))):
            for k in range(0, 5):
                try:
                    image = imread('Jacquard/Jacquard_Dataset_{}/{}/{}_{}_RGB.png'.format(i, dir_name, k, dir_name))
                    # image = image[start_x:end_x, start_y:end_y, :]
                    f = open('Jacquard/Jacquard_Dataset_{}/{}/{}_{}_grasps.txt'.format(i, dir_name, k, dir_name), 'r')
                    points = f.readlines()
                    old_shape = image.shape
                    image = cv2.resize(image, (224, 224))
                    new_shape = image.shape
                    x_ratio, y_ratio = new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]
                    points = [str(x_ratio * (float(point.split(';')[0]))) + ' ' +
                              str(x_ratio * (float(point.split(';')[1]))) + ' ' +
                              str(float(point.split(';')[2])) + ' ' +
                              str(x_ratio * float(point.split(';')[3])) + ' ' +
                              str(x_ratio * float(point.split(';')[4]))
                              + '\n' for point in points]
                    f.close()
                    f = open('processed_data_jacquard/pcd0{}cpos.txt'.format(j), 'w')
                    f.writelines(points)
                    imsave('processed_data_jacquard/pcd0{}r.png'.format(j), image)
                    j += 1
                except:
                    pass
                    # print('{}_{}_RGB.png does not exist'.format(k, dir_name))


def zoom_on_data_bis(viz=False):
    j = 0
    start_x, end_x, start_y, end_y = 154, 475, 61, 538
    for k in range(1, 11):
        for i in range(k*100, (k+1)*100):
            try:
                image = imread('folder/0{}/pcd0{}r.png'.format(k, i))
                image = image[start_x:end_x, start_y:end_y, :]
                rect = bounding_box_detector(image, viz=viz)
                if rect is not None:
                    image = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                    f = open('folder/0{}/pcd0{}cpos.txt'.format(k, i), 'r')
                    points = f.readlines()
                    old_shape = image.shape
                    image = cv2.resize(image, (224, 224))
                    new_shape = image.shape
                    x_ratio, y_ratio = new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]
                    f.close()

                    points = [str(x_ratio*(float(point.split(' ')[0])-start_y-rect[0]))+' '+str(x_ratio*(float(point.split(' ')[1])-start_x-rect[1]))+'\n' for point in points]

                    if viz:
                        rectangles = process(points)
                        image_good = draw_rectangle(rectangles, image)
                        plt.imshow(image_good)
                        plt.title('Good grasping rectangles')
                        plt.show()
                    if not True in ['nan' in pt for pt in points]:
                        f = open('processed_data/pcd0{}cpos.txt'.format(j), 'w')
                        f.writelines(points)
                        imsave('processed_data/pcd0{}r.png'.format(j), image)
                        j += 1
            except:
                print("Oops!", sys.exc_info()[0], "occured.")
                pass


def bboxes_to_grasps(box):
    # bboxes au format [x0, y0, x1, y1, x2, y2, x3, y3]
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    x = (box[0] + (box[4] - box[0])/2)
    y = (box[1] + (box[5] - box[1])/2)

    if box[0] == box[2]:
        tan = 30
    else:
        tan = -(box[3] - box[1]) / (box[2] - box[0])
    tan = max(-11, min(tan, 11))
    w = np.sqrt(np.power((box[2] - box[0]), 2) + np.power((box[3] - box[1]), 2))
    h = np.sqrt(np.power((box[6] - box[0]), 2) + np.power((box[7] - box[1]), 2))
    angle = np.arctan(tan) * 180/np.pi
    return x, y, angle, h, w


def grasp_to_bbox(grasp):
    # x, y, tan, h, w = grasp
    #theta = np.arctan(tan)
    print(grasp)
    x, y, theta, h, w = tuple(grasp)
    theta = theta * np.pi/180
    edge1 = [x - w/2*np.cos(theta) + h/2*np.sin(theta), y + w/2*np.sin(theta) + h/2*np.cos(theta)]
    edge2 = [x + w/2*np.cos(theta) + h/2*np.sin(theta), y - w/2*np.sin(theta) + h/2*np.cos(theta)]
    edge3 = [x + w/2*np.cos(theta) - h/2*np.sin(theta), y - w/2*np.sin(theta) - h/2*np.cos(theta)]
    edge4 = [x - w/2*np.cos(theta) - h/2*np.sin(theta), y + w/2*np.sin(theta) - h/2*np.cos(theta)]
    return [edge1, edge2, edge3, edge4]


def create_preprocessing_data(path, split_percent=0.8, viz=False):
    X_train, Y_train, X_test, Y_test, X_test_mse, Y_test_mse = [], [], [], [], [], []
    A = [file for file in os.listdir(path) if ('.png' in file)]
    for image_name in A:
        image = imread(path+'/'+image_name)
        number = image_name.rstrip('r.png')
        f = open(path + '/' + number + 'cpos.txt')
        grasp_points = f.readlines()
        y_temp = []
        trainORtest = np.random.random() < split_percent

        if trainORtest:                                     # Dans le cas où c'est dans le train ==> injection
            # for nb_grasp in range(len(grasp_points)//4):  # Si on ne veut pas être injectif
            for nb_grasp in range(1):  # Si on veut être injectif
                box = []
                for i in range(4):
                    temp_grasp = list(map(float, grasp_points[nb_grasp*4+i].split(" ")))
                    box.append(temp_grasp[0])
                    box.append(temp_grasp[1])
                grasp_param = bboxes_to_grasps(box)
                if viz:
                    plt.imshow(image)
                    plt.scatter(grasp_param[0], grasp_param[1])
                    plt.show()
                nan_check = [grasp == grasp for grasp in grasp_param]
                if False not in nan_check:
                    ## We put in train set several couple (X,Y) for the same X
                    X_train.append(image)
                    Y_train.append(grasp_param)

        else:                                               # Sinon dans le test, on prend tous les rectangles possibles
            ############### Dans le test MSE on n'en prend qu'un seul ###############
            # for nb_grasp in range(1):  # Si on veut être injectif
            #     box = []
            #     for i in range(4):
            #         temp_grasp = list(map(float, grasp_points[nb_grasp*4+i].split(" ")))
            #         box.append(temp_grasp[0])
            #         box.append(temp_grasp[1])
            #     grasp_param = bboxes_to_grasps(box)
            #     nan_check = [grasp == grasp for grasp in grasp_param]
            #     if False not in nan_check:
            #         ## We put in train set several couple (X,Y) for the same X
            #         X_test_mse.append(image)
            #         Y_test_mse.append(grasp_param)

            for nb_grasp in range(len(grasp_points)//4):
                box = []
                for i in range(4):
                    temp_grasp = list(map(float, grasp_points[nb_grasp * 4 + i].split(" ")))
                    box.append(temp_grasp[0])
                    box.append(temp_grasp[1])
                grasp_param = bboxes_to_grasps(box)
                ## We put in test set one couple (X, [Y1,..,Yn])
                y_temp.append(grasp_param)
            X_test.append(image)
            Y_test.append(y_temp)

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def all_data_on_test_format(path):
    X, Y = [], []
    A = [file for file in os.listdir(path) if ('.png' in file)]
    for image_name in A:
        image = imread(path + '/' + image_name)
        number = image_name.rstrip('r.png')
        f = open(path + '/' + number + 'cpos.txt')
        grasp_points = f.readlines()
        y_temp = []

        for nb_grasp in range(len(grasp_points)//4):
            box = []
            for i in range(4):
                temp_grasp = list(map(float, grasp_points[nb_grasp * 4 + i].split(" ")))
                box.append(temp_grasp[0])
                box.append(temp_grasp[1])
            grasp_param = bboxes_to_grasps(box)
            ## We put in test set one couple (X, [Y1,..,Yn])
            y_temp.append(grasp_param)
        X.append(image)
        Y.append(y_temp)

    return np.array(X), np.array(Y)


def augmentation(X, Y):
    new_X, new_Y = [], []
    for x, y_list in zip(X, Y):
        new_X.append(x)
        new_Y.append(y_list)
        fliplr_x = np.fliplr(np.copy(x))
        y_temp = []
        for y in y_list:
            bb = np.array(grasp_to_bbox(y))
            bb[:, 0] = 224 - bb[:, 0]
            grasp = bboxes_to_grasps(bb.ravel())
            y_temp.append(grasp)
        new_X.append(fliplr_x)
        new_Y.append(y_temp)
        #####################################
        ## Ajouter la luminosité plus tard ##
        #####################################
        flipud_x = np.flipud(np.copy(x))
        y_temp = []
        for y in y_list:
            bb = np.array(grasp_to_bbox(y))
            bb[:, 1] = 224 - bb[:, 1]
            grasp = bboxes_to_grasps(bb.ravel())
            y_temp.append(grasp)
        new_X.append(flipud_x)
        new_Y.append(y_temp)
        #####################################
        ## Ajouter la luminosité plus tard ##
        #####################################
        flipudlr_x = np.fliplr(np.flipud(np.copy(x)))
        y_temp = []
        for y in y_list:
            bb = np.array(grasp_to_bbox(y))
            bb[:, 0], bb[:, 1] = 224 - bb[:, 0], 224 - bb[:, 1]
            grasp = bboxes_to_grasps(bb.ravel())
            y_temp.append(grasp)
        new_X.append(flipudlr_x)
        new_Y.append(y_temp)

    return np.array(new_X), np.array(new_Y)


def bounding_box_detector(image, viz=False):
    # image = image[154:475, 61:538]
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thres, thresh_output = cv2.threshold(img_grey, 150, 255, cv2.THRESH_BINARY)
    kernel = -np.ones((5, 5), np.uint8)
    thresh_output = cv2.bitwise_not(thresh_output)
    thresh_output = cv2.dilate(thresh_output, kernel, iterations=10)
    thresh_output = cv2.bitwise_not(thresh_output)
    cnts, hier = cv2.findContours(thresh_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        rect = list(cv2.boundingRect(cnts[1]))
        L = max((rect[2:]))
        rect[0] = max(int(rect[0] - L / 2 + rect[2] // 2), 0)
        rect[1] = max(int(rect[1] - L / 2 + rect[3] // 2), 0)
        rect[2] = rect[3] = L
    except:
        print('Error while finding the contour')
        return None
    if viz:
        cv2.rectangle(image, tuple(rect), (0, 255, 0), 5)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(thresh_output)
        plt.show()
    return rect


# def create_data_format_from_jacquard(folder):


if __name__=="__main__":
    vizualize_jacquard()

    X, Y = all_data_on_test_format('processed_data_jacquard')
    np.save('prepared_data/all_X_jacquard_augmented_test_format.npy', X)
    np.save('prepared_data/all_Y_jacquard_augmented_test_format.npy', Y)
    pass

    # zoom_on_data_bis(viz=False)

    # vizualise('processed_data')
    # X, Y = all_data_on_test_format('processed_data')
    ################ Create augmented data ###############
    # X, Y = all_data_on_test_format('processed_data')
    # new_X, new_Y = augmentation(X, Y)
    # np.save('prepared_data/all_X_augmented_test_format.npy', new_X)
    # np.save('prepared_data/all_Y_augmented_test_format.npy', new_Y)
    # X, Y = np.load('prepared_data/all_X_augmented_test_format.npy', allow_pickle=True),\
    #        np.load('prepared_data/all_Y_augmented_test_format.npy', allow_pickle=True)
    # vizualise_all(X, Y)
    # for i in range(101, 200):
    #     X = imread('folder/0{}/pcd0{}r.png'.format(i))
    #     my_object_detector = load_object_detector()
    #     object_detector(my_object_detector, X)