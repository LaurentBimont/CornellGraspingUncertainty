from ConfidLossNet import lossnet, confidnet
from MCD_uncertainty import make_classification
from model import model_resnet, resnet_model, model_vgg, concrete_Dropout_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.widgets import Slider, Button, RadioButtons

from sklearn.metrics import auc
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json



if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    K.set_session(tf.Session(config=config))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.enable_eager_execution(config)


def ROC_curve(good_Y, bad_Y, viz=False):
    threshold_list = np.linspace(1.01*min(np.concatenate((good_Y, bad_Y))), 0.99*max(np.concatenate((good_Y, bad_Y))), 100)
    X, Y = [], []
    for thresh in threshold_list:
        TP, FP = len(bad_Y[bad_Y > thresh]), len(good_Y[good_Y > thresh])
        FN, TN = len(bad_Y[bad_Y < thresh]), len(good_Y[good_Y < thresh])
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        X.append(FPR)
        Y.append(TPR)
    if viz:
        plt.plot(X, Y, color='red')
        plt.show()
    return X,  Y


def PR_curve(good_Y, bad_Y, viz=False):
    threshold_list = np.linspace(1.01*min(np.concatenate((good_Y, bad_Y))), 0.99*max(np.concatenate((good_Y, bad_Y))), 100)
    X, Y = [], []
    for thresh in threshold_list:
        TP, FP = len(bad_Y[bad_Y >= thresh]), len(good_Y[good_Y > thresh])
        FN, TN = len(bad_Y[bad_Y <= thresh]), len(good_Y[good_Y < thresh])
        grasping_success = TN / (TN + FN + 0.001)
        # recall = TN / (FN + TN)
        FPER = FP / (FP + TN)
        Y.append(grasping_success)
        X.append(FPER)
        # gc, gu = len(good_Y[good_Y <= thresh]), len(good_Y[good_Y > thresh])
        # bc, bu = len(bad_Y[bad_Y <= thresh]), len(bad_Y[bad_Y > thresh])
        # Y.append(gc / (gc + bc))
        # X.append(bu / (gc + bu))
    if viz:
        plt.plot(X, Y, color='red')
        plt.show()
    return X, Y


def plot_uncertainty_result(good_Y, bad_Y, viz=True, thresh=None, reg=None):
    if reg is None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
        ax4.scatter(good_Y, reg[0], color='green')
        ax4.scatter(bad_Y, reg[1], color='red')
        ax4.set_xlabel('True Value')
        ax4.set_ylabel('Predicted Value')
        ax4.set_title('MSE prediction repartion\n Pearson coef {}'.format(round(100*reg[2][0])/100))

    unc_max, unc_min = max(np.concatenate((bad_Y, good_Y)))*0.99, min(np.concatenate((bad_Y, good_Y)))*1.01
    bins = np.linspace(0, unc_max, 20)
    if thresh is None:
        axamp = plt.axes([0.4, 0.95, 0.2, 0.03])
        seuil = Slider(axamp, 'Threshold', unc_min, unc_max, valinit=2*min(np.concatenate((good_Y, bad_Y))))
        my_thresh = seuil.val
    else:
        my_thresh = thresh

    count = np.array([-1 for i in range(len(bad_Y))])

    ax1.hist(bad_Y, color='r', bins=bins, weights=count, label='Bad prediction')
    ax1.hist(good_Y, color='g', bins=bins, label='Good prediction')

    ax1.plot([my_thresh, my_thresh], [-100, 300], '--', color='black')
    ax1.set_xlabel('Uncertainty metric')
    ax1.set_ylabel('Number of occurences')
    ax1.set_xlim(min(bins), max(bins))
    ax1.set_ylim(-80, 200)
    ax1.set_title('Distribution of bad/good predictions')
    ax1.legend()
    # ax1.title('Histogram of good/bad predictions')

    #### Axe 2 ####

    uncertainty_table = [[len(bad_Y[bad_Y > my_thresh]), len(good_Y[good_Y > my_thresh])],
                         [len(bad_Y[bad_Y < my_thresh]), len(good_Y[good_Y < my_thresh])]]

    # ax2.title('Uncertainty matrix')
    unc_mat = ax2.imshow(uncertainty_table, cmap='Greens')
    unc_number = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            text_numb = ax2.text(j, i, uncertainty_table[i][j], horizontalalignment='center', color='black', fontsize=15)
            unc_number[i][j] = text_numb
    ax2.text(-0.8, 0.5, 'Action', rotation=90, fontsize='large', horizontalalignment='center',
             verticalalignment='center')
    ax2.text(-0.6, 0, 'Ask (failure)', rotation=90, horizontalalignment='center', verticalalignment='center')
    ax2.text(-0.6, 1, 'Act (no failure)', rotation=90, horizontalalignment='center', verticalalignment='center')

    ax2.text(.5, +1.8, 'True', fontsize='large', horizontalalignment='center', verticalalignment='center')
    ax2.text(0, 1.6, 'Failure', horizontalalignment='center', verticalalignment='center')
    ax2.text(1, 1.6, 'No failure', horizontalalignment='center', verticalalignment='center')
    # plt.axis('off')
    # plt.grid()
    ax2.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')    # ax2.yticks([])
    ax2.plot([-0.7, 1.5], [0.5, 0.5], c='black')
    ax2.plot([0.5, 0.5], [-0.5, 1.7], c='black')
    # Contour
    ax2.plot([1.5, 1.5], [-0.5, 1.9], c='black')
    ax2.plot([-.5, -.5], [-0.5, 1.9], c='black')
    ax2.plot([-0.9, 1.5], [-0.5, -0.5], c='black')
    ax2.plot([-0.9, 1.5], [1.5, 1.5], c='black')
    # Contour de action
    ax2.plot([-0.5, 1.5], [1.7, 1.7], c='black')
    ax2.plot([-0.5, 1.5], [1.9, 1.9], c='black')
    # Contour de la prédiction
    ax2.plot([-0.9, -0.9], [-0.5, 1.5], c='black')
    ax2.plot([-0.7, -0.7], [-0.5, 1.5], c='black')
    ax2.set_title('Uncertainty matrix')


    # ROC curve
    X, Y = ROC_curve(good_Y, bad_Y)
    TP, FP, FN, TN = uncertainty_table[0][0], uncertainty_table[0][1], uncertainty_table[1][0], uncertainty_table[1][1]
    # bc, bu = len(bad_Y[bad_Y < my_thresh]), len(bad_Y[bad_Y > my_thresh])

    # FPR = FP/(FP+TN)
    # TPR = TP/(TP+FN)
    # precision = TP/(TP+FP)

    ratio_ROC = FP/(FP+TN)

    ax3.plot(100*np.array(X), 100*np.array(Y), color='blue', linewidth=2, label="ROC")
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax3.xaxis.set_major_formatter(xticks)
    ax3.yaxis.set_major_formatter(xticks)
    ax3.text(50, 40, 'AUROC : {}%'.format(round(1000*auc(X, Y))/10), fontsize=15)

    # PR curve
    X, Y = PR_curve(good_Y, bad_Y)

    ratio_PR = TP/(TP+FP)
    ax3.plot(100 * np.array(X), 100 * np.array(Y), color='orange', linewidth=2, label="PR")
    fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax3.xaxis.set_major_formatter(xticks)
    ax3.yaxis.set_major_formatter(xticks)
    ax3.text(50, 20, 'AUPR : {}%'.format(round(1000 * auc(X, Y)) / 10), fontsize=15)

    # Baseline
    ax3.plot([0, 100], [0, 100], color='black')
    ax3.set_ylim([0, 100])
    ax3.set_xlim([0, 100])
    # ax3.plot(X, Y_1, color='green', label='False positive rate')
    ax3.plot([100*ratio_ROC, 100*ratio_ROC], [0, 100], '--', color='blue')
    ax3.plot([100 * ratio_PR, 100 * ratio_PR], [0, 100], '--', color='orange')
    ax3.set_title('ROC curve')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    plt.tight_layout()

    def update(value):
        uncertainty_table = [[len(good_Y[good_Y < seuil.val]), len(good_Y[good_Y > seuil.val])],
                             [len(bad_Y[bad_Y < seuil.val]), len(bad_Y[bad_Y > seuil.val])]]

        uncertainty_table = [[len(bad_Y[bad_Y > seuil.val]), len(good_Y[good_Y > seuil.val])],
                             [len(bad_Y[bad_Y < seuil.val]), len(good_Y[good_Y < seuil.val])]]

        unc_mat.set_data(uncertainty_table)
        for i in range(2):
            for j in range(2):
                unc_number[i][j].set_text(uncertainty_table[i][j])
        ax1.lines.pop(0)  # remove previous line plot
        ax1.plot([seuil.val, seuil.val], [-100, 300], '--', color='black')

        TP, FP, FN, TN = uncertainty_table[0][0], uncertainty_table[0][1], uncertainty_table[1][0], \
                         uncertainty_table[1][1]

        ratio_ROC = FP / (FP + TN)
        ratio_PR = FP / (FP + TN)

        ax3.lines.pop(-1)
        ax3.plot([100*ratio_ROC, 100*ratio_ROC], [0, 100], '--', color='blue')
        ax3.lines.pop(-1)
        ax3.plot([100 * ratio_PR, 100 * ratio_PR], [0, 100], '--', color='orange')

    if thresh is None:
        seuil.on_changed(update)
    if viz:
        plt.show()
    else:
        return f


def make_film(good_Y, bad_Y):
    seuil_liste = np.linspace(0, 1, 1000)
    dd = 0
    for thresh in seuil_liste:
        f = plot_uncertainty_result(good_Y, bad_Y, viz=False, thresh=thresh)
        f.savefig('video/{}.png'.format(dd))
        plt.close(f)
        dd += 1
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #
    #
    # line_ani = animation.FuncAnimation(f, func=lambda x:x ,frames=ims, interval=50, blit=True)
    # line_ani.save('lines.mp4', writer=writer)


def save_result_curves(good_Y, bad_Y, model_name):
    X, Y = PR_curve(good_Y, bad_Y)
    print('Resultats/X_PR_{}.npy'.format(model_name))
    np.save('Resultats/X_PR_{}.npy'.format(model_name), X)
    np.save('Resultats/Y_PR_{}.npy'.format(model_name), Y)
    X, Y = ROC_curve(good_Y, bad_Y)
    np.save('Resultats/X_ROC_{}.npy'.format(model_name), X)
    np.save('Resultats/Y_ROC_{}.npy'.format(model_name), Y)


######## Plot #######
def plot_ensemble(Y, res):
    mean_var = Y[:, :2].var(axis=1)
    good_pred = np.where(res == 1)
    bad_pred = np.where(res == 0)
    good_Y = mean_var[good_pred]
    bad_Y = mean_var[bad_pred]
    save_result_curves(good_Y, bad_Y, 'ensemble1')
    plot_uncertainty_result(good_Y, bad_Y)

def plot_proper_ensemble(Y, res):
    var = Y[:, :, 5:]
    mean = Y[:, :, :5]

    var_plus_mean = var + np.square(mean)
    var_mean = var_plus_mean.mean(axis=0) - mean.mean(axis=0)
    mean_var = var_mean.mean(axis=-1)
    good_pred = np.where(res == 1)
    bad_pred = np.where(res == 0)
    max_value = np.max(mean_var) +1
    good_Y = max_value  - mean_var[good_pred]
    bad_Y = max_value - mean_var[bad_pred]
    save_result_curves(good_Y, bad_Y, 'ensemble+proper_score')
    plot_uncertainty_result(good_Y, bad_Y)


def plot_proper_score(Y, res):
    mean_var = Y[:, 5:].mean(axis=-1)
    good_pred = np.where(res == 1)
    bad_pred = np.where(res == 0)
    good_Y = 100-mean_var[good_pred]
    bad_Y = 100-mean_var[bad_pred]
    save_result_curves(good_Y, bad_Y, 'proper_score1')
    plot_uncertainty_result(good_Y, bad_Y)


def plot_bayesian(lossnet_data, model, model_name, recompute=True):
    X, res = lossnet_data[:, 0], lossnet_data[:, 4]
    X = np.array([np.array(x) for x in X])

    _, X_test, _, res = train_test_split(X, res, test_size=0.33, random_state=42)

    if recompute:

        # model = load_my_model('saved_model/model_arch_{}.json'.format(model_name), 'saved_model/my_model_weights_{}.h5'.format(model_name))
        f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

        X_test = X_test.astype(np.uint8)
        T_bayesian_draw = []
        for i in tqdm(range(X_test.shape[0]//10+1)):
            temp = np.array([f((X_test[i*10:min((i+1)*10, X_test.shape[0])], 1))[0] for i in range(10)])
            lala = temp.var(axis=0)
            T_bayesian_draw.extend(lala)

        var = np.array(T_bayesian_draw)

        np.save('uncertainty/MC_dropout_var_{}.npy'.format(model_name), var)
    else:
        var = np.load('uncertainty/MC_dropout_var_{}.npy'.format(model_name))

    mean_var = var[:, :].mean(axis=1)
    good_pred = np.where(res == 1)
    bad_pred = np.where(res == 0)
    good_Y = mean_var[good_pred]
    bad_Y = mean_var[bad_pred]
    save_result_curves(good_Y, bad_Y, model_name)

    plot_uncertainty_result(good_Y, bad_Y)


def plot_lossnet(lossnet_data, model_name, Restrainable=False):
    if not Restrainable:
        X, MSE, result_test = lossnet_data[:, 1], lossnet_data[:, 3], lossnet_data[:, 4]
        X = np.array([np.array(x) for x in X])
    else:
        X, MSE, result_test = lossnet_data[:, 1], lossnet_data[:, 3], lossnet_data[:, 4]
        X = np.array([np.array(x) for x in X])

    _, X_test, _, _ = train_test_split(X, MSE, test_size=0.33, random_state=42)
    _, MSE, _, result_test = train_test_split(MSE, result_test, test_size=0.33, random_state=42)

    # model = lossnet()
    # model.fit(X_test, MSE, batch_size=100, epochs=1)
    json_file = open('saved_model/model_arch_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)


    model.load_weights('saved_model/checkpoint_{}.h5'.format(model_name))

    MSE_pred = model.predict(X_test)
    good_pred = np.where(result_test == 1)
    bad_pred = np.where(result_test == 0)

    good_Y, bad_Y = (np.array([MSE_pred[good_pred].squeeze()]))[0], (np.array([MSE_pred[bad_pred].squeeze()]))[0]
    pearson_coef = pearsonr(MSE, MSE_pred.reshape((-1)))

    save_result_curves(good_Y, bad_Y, 'Lossnet1')
    plot_uncertainty_result(good_Y, bad_Y, reg=[np.array(MSE[good_pred]), np.array(MSE[bad_pred]), pearson_coef])

    plt.savefig('{}.png'.format(model_name), dpi=600)


def plot_confidnet(lossnet_data, model_name, Restrainable=False):
    if not Restrainable:
        X, res = lossnet_data[:, 1], lossnet_data[:, 4]
        X = np.array([np.array(x) for x in X])
    else:
        X, res = lossnet_data[:, 0], lossnet_data[:, 4]
        X = np.array([np.array(x) for x in X])

    _, X_test, _, res = train_test_split(X, res, test_size=0.33, random_state=42)

    # model = confidnet(Restrainable=Restrainable)
    # model.fit(X_test[:1], res[:1], batch_size=1, epochs=1)
    json_file = open('saved_model/model_arch_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('saved_model/checkpoint_{}.h5'.format(model_name))
    # model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))
    Y_pred = model.predict(X_test, batch_size=20)
    good_pred = np.where(res == 1)
    bad_pred = np.where(res == 0)
    # print(good_pred, bad_pred, Y_pred[bad_pred], len(result_test))

    good_Y, bad_Y = (1-np.array([Y_pred[good_pred].squeeze()]))[0], (1-np.array([Y_pred[bad_pred].squeeze()]))[0]

    # make_film(good_Y, bad_Y)
    # ROC_curve(good_Y, bad_Y)
    save_result_curves(good_Y, bad_Y, 'ConfidNet1')

    plot_uncertainty_result(good_Y, bad_Y)

if __name__=='__main__':
    # MSE, mean_var, T_bayesian_draw = np.load('uncertainty/mse.npy'), np.load('uncertainty/mean_var.npy'), np.load('uncertainty/T_bayesian_draw.npy')
    # result_test = np.load('uncertainty/result_test.npy')
    # plot_bayesian_uncertainty(MSE, T_bayesian_draw, result_test)

    result_test = np.load('uncertainty/good_bad_lossnet.npy')
    # X_train, Y_train, X_test, Y_test = np.load('prepared_data/X_train.npy', allow_pickle=True), \
    #                                    np.load('prepared_data/Y_train.npy', allow_pickle=True), \
    #                                    np.load('prepared_data/X_test.npy', allow_pickle=True), \
    #                                    np.load('prepared_data/Y_test.npy', allow_pickle=True)

    lossnet_data = np.load('uncertainty/lossnet_data_aug_VGG16_6.npy', allow_pickle=True)

    ######## Ensemble + proper
    # Y = np.load('uncertainty/deep_ensemble_proper_output_5_VGG16.npy')
    # res = np.load('uncertainty/deep_ensemble_proper_result_5_VGG16.npy')
    # plot_proper_ensemble(Y, res)

    ######## Ensemble
    # Y = np.load('uncertainty/deep_ensemble_output_5_VGG16.npy')
    # res = np.load('uncertainty/deep_ensemble_result_5_VGG16.npy')
    # plot_ensemble(Y, res)

    ######## Proper Score
    # Y = np.load('uncertainty/proper_score_output_1.npy')
    # res = np.load('uncertainty/proper_score_result_1.npy')
    # plot_proper_score(Y, res)

    ######## Flipout



    ######## MonteCarlo Net
    # lossnet_data = np.load('uncertainty/lossnet_data_aug_VGG16_6.npy', allow_pickle=True)
    # model = model_vgg(bayesian=True)
    # model.fit(np.random.random((100, 224, 224, 3)), np.random.random((100, 5)))
    # model.load_weights('saved_model/checkpoint_deep_ensemble_VGG16_6.h5')
    # plot_bayesian(lossnet_data, model, model_name='VGG16_6', recompute=False)

    ######### Concrete Monte Carlo
    # lossnet_data = np.load('uncertainty/lossnet_data_aug_VGG16_6.npy', allow_pickle=True)
    # model = concrete_Dropout_model()
    # model.load_weights('saved_model/checkpoint_ConcreteDropout_1.h5')
    # plot_bayesian(lossnet_data, model, model_name='ConcreteDropout', recompute=False)

    # ######## Loss Net ##########
    plot_lossnet(lossnet_data, model_name='LossNet_7', Restrainable=False)

    ######## Confid Net ########
    plot_confidnet(lossnet_data, model_name='ConfidNet_7', Restrainable=False)
