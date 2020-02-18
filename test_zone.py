import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

K.set_session(tf.Session(config=config))


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

import numpy as np
import tensorflow_probability as tfp


def neg_log_likelihood(y_true, y_pred):
    my_mean = y_pred[:, :5]
    my_var = y_pred[:, 5:]
    my_mean_temp = K.repeat(my_mean, K.shape(y_true)[1])
    my_var = (K.log(1 + K.exp(my_var)))
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) +\
    #        0.5*K.log(2*np.pi)
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) + 0.5*K.log(2*np.pi)   K.min(K.square(my_mean_temp-y_true), axis=1)/
    numerateur = K.min(K.square(my_mean_temp - y_true), axis=1)
    denominateur = 2 * K.square(my_var + 1e-5)
    print(K.get_value(numerateur), K.get_value(denominateur))
    print(K.get_value(numerateur/denominateur))
    return K.mean(numerateur/denominateur)

y_true = -1*tf.random.uniform((100, 20, 5))
y_pred = 10000*tf.ones((100, 10))

print(K.get_value(neg_log_likelihood(y_true, y_pred)))