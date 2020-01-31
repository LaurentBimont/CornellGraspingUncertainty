from ConfidLossNet import lossnet, confidnet
from MCD_uncertainty import make_classification
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.widgets import Slider, Button, RadioButtons

from sklearn.metrics import auc
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split


def ROC_curve(good_Y, bad_Y, viz=False):
    threshold_list = np.linspace(0, max(np.concatenate((good_Y, bad_Y))), 100)
    X, Y, Y_1 = [], [], []
    for thresh in threshold_list:
        gc, gu = len(good_Y[good_Y < thresh]), len(good_Y[good_Y > thresh])
        bc, bu = len(bad_Y[bad_Y < thresh]), len(bad_Y[bad_Y > thresh])
        Y.append(1-(gu/(gc+bu+gu)))
        X.append(bc/(bc+bu))
        Y_1.append(gc/(gc+gu))
    if viz:
        plt.plot(X, Y, color='red')
        plt.plot(X, Y_1, color='green')
        plt.show()
    return X, Y, Y_1


def plot_uncertainty_result(good_Y, bad_Y, viz=True, thresh=None):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    unc_max = max(np.concatenate((bad_Y, good_Y)))
    bins = np.linspace(0, unc_max, 20)
    if thresh is None:
        axamp = plt.axes([0.4, 0.95, 0.2, 0.03])
        seuil = Slider(axamp, 'Threshold', 0., unc_max, valinit=0.2*unc_max)
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
    uncertainty_table = [[len(good_Y[good_Y < my_thresh]), len(good_Y[good_Y > my_thresh])],
                         [len(bad_Y[bad_Y < my_thresh]), len(bad_Y[bad_Y > my_thresh])]]
    # ax2.title('Uncertainty matrix')
    unc_mat = ax2.imshow(uncertainty_table, cmap='Greens')
    unc_number = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            text_numb = ax2.text(j, i, uncertainty_table[i][j], horizontalalignment='center', color='black', fontsize=15)
            unc_number[i][j] = text_numb
    ax2.text(-0.8, 0.5, 'Prediction', rotation=90, fontsize='large', horizontalalignment='center',
             verticalalignment='center')
    ax2.text(-0.6, 0, 'Good', rotation=90, horizontalalignment='center', verticalalignment='center')
    ax2.text(-0.6, 1, 'Bad', rotation=90, horizontalalignment='center', verticalalignment='center')

    ax2.text(.5, +1.8, 'Action', fontsize='large', horizontalalignment='center', verticalalignment='center')
    ax2.text(0, 1.6, 'Act', horizontalalignment='center', verticalalignment='center')
    ax2.text(1, 1.6, 'Ask', horizontalalignment='center', verticalalignment='center')
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


    X, Y_1, Y_2 = ROC_curve(good_Y, bad_Y)
    bc, bu = len(bad_Y[bad_Y < my_thresh]), len(bad_Y[bad_Y > my_thresh])
    ratio_ROC = bc/(bc+bu)
    ax3.plot(100*np.array(X), 100*np.array(Y_2), color='red')
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax3.xaxis.set_major_formatter(xticks)
    ax3.yaxis.set_major_formatter(xticks)


    ax3.text(60, 40, 'AUC : {}%'.format(round(1000*auc(X, Y_2))/10), fontsize=15)
    ax3.plot([0, 100], [0, 100], color='black')
    ax3.set_ylim([0, 100])
    ax3.set_xlim([0, 100])
    # ax3.plot(X, Y_1, color='green', label='False positive rate')
    ax3.plot([100*ratio_ROC, 100*ratio_ROC], [0, 100], '--', color='black')
    ax3.set_title('ROC curve')
    ax3.set_xlabel('Bad sure rate')
    ax3.set_ylabel('Good sure rate')
    plt.tight_layout()
    def update(value):
        uncertainty_table = [[len(good_Y[good_Y < seuil.val]), len(good_Y[good_Y > seuil.val])],
                             [len(bad_Y[bad_Y < seuil.val]), len(bad_Y[bad_Y > seuil.val])]]

        unc_mat.set_data(uncertainty_table)
        for i in range(2):
            for j in range(2):
                unc_number[i][j].set_text(uncertainty_table[i][j])
        ax1.lines.pop(0)  # remove previous line plot
        ax1.plot([seuil.val, seuil.val], [-100, 300], '--', color='black')

        bc, bu = len(bad_Y[bad_Y < seuil.val]), len(bad_Y[bad_Y > seuil.val])
        ratio_ROC = bc / (bc + bu)
        ax3.lines.pop(-1)
        ax3.plot([100*ratio_ROC, 100*ratio_ROC], [0, 100], '--', color='black')
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


######## Plot #######
def plot_bayesian_uncertainty(MSE, T_bayesian_draw, result_test):

    good_pred = np.where(result_test == 1)
    bad_pred = np.where(result_test == 0)

    mean_var = T_bayesian_draw[:, :, :2].var(axis=0).mean(axis=1)
    #
    # plt.scatter(MSE[good_pred], mean_var[good_pred], color='g')
    # plt.scatter(MSE[bad_pred], mean_var[bad_pred], color='r')
    #
    # plt.xlabel('Mean Squared Error')
    # plt.ylabel('Variance')
    # plt.title('Corrélation entre MSE et variance (Pearson coefficient : {})'.format(round(100*pearsonr(MSE, mean_var)[0])/100))
    # plt.show()

    plt.subplot(1, 2, 1)
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    count = [-1 for i in range(len(mean_var[bad_pred]))]
    seuil = min(mean_var[bad_pred].squeeze())
    plt.hist([mean_var[good_pred].squeeze()], color='g', bins=bins)
    plt.hist([mean_var[bad_pred].squeeze()], color='r', weights=count, bins=bins)
    plt.xlabel('Predicted Class')
    plt.ylabel('Number of occurences')
    plt.plot([seuil, seuil], [-20, 20], linestyle='--', linewidth=2, color='black')
    plt.title('Histogram of good/bad predictions')
    plt.subplot(1, 2, 2)

    good_pred, bad_pred = mean_var[good_pred], mean_var[bad_pred]
    seuil = max(bad_pred)

    uncertainty_table = [[len(good_pred[good_pred > seuil]), len(good_pred[good_pred < seuil])],
                         [len(bad_pred[bad_pred > seuil]), len(bad_pred[bad_pred < seuil])]]
    plt.title('Uncertainty matrix')
    plt.imshow(uncertainty_table, cmap='Greens')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, uncertainty_table[i][j], horizontalalignment='center')
    plt.text(-0.8, 0.5, 'Prediction', rotation=90, fontsize='large', horizontalalignment='center',
             verticalalignment='center')
    plt.text(-0.6, 0, 'Good', rotation=90, horizontalalignment='center', verticalalignment='center')
    plt.text(-0.6, 1, 'Bad', rotation=90, horizontalalignment='center', verticalalignment='center')

    plt.text(.5, +1.8, 'Action', fontsize='large', horizontalalignment='center', verticalalignment='center')
    plt.text(0, 1.6, 'Act', horizontalalignment='center', verticalalignment='center')
    plt.text(1, 1.6, 'Ask', horizontalalignment='center', verticalalignment='center')
    # plt.axis('off')
    # plt.grid()
    plt.xticks([])
    plt.yticks([])
    plt.plot([-0.7, 1.5], [0.5, 0.5], c='black')
    plt.plot([0.5, 0.5], [-0.5, 1.7], c='black')
    # Contour
    plt.plot([1.5, 1.5], [-0.5, 1.9], c='black')
    plt.plot([-.5, -.5], [-0.5, 1.9], c='black')
    plt.plot([-0.9, 1.5], [-0.5, -0.5], c='black')
    plt.plot([-0.9, 1.5], [1.5, 1.5], c='black')
    # Contour de action
    plt.plot([-0.5, 1.5], [1.7, 1.7], c='black')
    plt.plot([-0.5, 1.5], [1.9, 1.9], c='black')
    # Contour de la prédiction
    plt.plot([-0.9, -0.9], [-0.5, 1.5], c='black')
    plt.plot([-0.7, -0.7], [-0.5, 1.5], c='black')

    # plt.title('Uncertainty matrix (threshold : {})'.format(round(seuil)))
    plt.show()


def plot_lossnet(X_test, MSE, result_test):
    _, X_test, _, _ = train_test_split(X_test, MSE, test_size=0.33, random_state=42)
    _, MSE, _, result_test = train_test_split(MSE, result_test, test_size=0.33, random_state=42)
    model_name = 'lossNet_1'
    model = lossnet()
    model.fit(X_test, MSE, batch_size=100, epochs=1)

    model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))
    MSE_pred = model.predict(X_test)
    good_pred = np.where(result_test == 1)
    bad_pred = np.where(result_test == 0)

    good_Y, bad_Y = (np.array([MSE_pred[good_pred].squeeze()]))[0], (np.array([MSE_pred[bad_pred].squeeze()]))[0]

    plot_uncertainty_result(good_Y, bad_Y)


def plot_confidnet(X_test, res, result_test):
    _, X_test, _, _ = train_test_split(X_test, res, test_size=0.33, random_state=42)
    _, res, _, result_test = train_test_split(res, result_test, test_size=0.33, random_state=42)
    model_name = 'ConfidNet_aug'
    model = confidnet()
    model.fit(X_test, res, batch_size=100, epochs=1)
    model.load_weights('saved_model/my_model_weights_{}.h5'.format(model_name))
    Y_pred = model.predict(X_test)
    good_pred = np.where(result_test == 1)
    bad_pred = np.where(result_test == 0)
    # print(good_pred, bad_pred, Y_pred[bad_pred], len(result_test))

    good_Y, bad_Y = (1-np.array([Y_pred[good_pred].squeeze()]))[0], (1-np.array([Y_pred[bad_pred].squeeze()]))[0]


    # make_film(good_Y, bad_Y)
    # ROC_curve(good_Y, bad_Y)
    plot_uncertainty_result(good_Y, bad_Y)



if __name__=='__main__':
    MSE, mean_var, T_bayesian_draw = np.load('uncertainty/mse.npy'), np.load('uncertainty/mean_var.npy'), np.load('uncertainty/T_bayesian_draw.npy')
    result_test = np.load('uncertainty/result_test.npy')
    # plot_bayesian_uncertainty(MSE, T_bayesian_draw, result_test)

    result_test = np.load('uncertainty/good_bad_lossnet.npy')
    X_train, Y_train, X_test, Y_test = np.load('prepared_data/X_train.npy', allow_pickle=True), \
                                       np.load('prepared_data/Y_train.npy', allow_pickle=True), \
                                       np.load('prepared_data/X_test.npy', allow_pickle=True), \
                                       np.load('prepared_data/Y_test.npy', allow_pickle=True)

    lossnet_data = np.load('uncertainty/lossnet_data_1.npy', allow_pickle=True)
    print('taille du bordel', lossnet_data.shape)
    X, MSE, res = lossnet_data[:, 1], lossnet_data[:, 3], lossnet_data[:, 4]
    X = np.array([np.array(x) for x in X])
    print('Taille du X', X.shape)

    plot_lossnet(X, MSE, res)

    ######## Confid Net ########
    X, res = lossnet_data[:, 1], lossnet_data[:, 4]
    X = np.array([np.array(x) for x in X])

    plot_confidnet(X, res, res)
