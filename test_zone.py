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


model = tf.keras.Sequential([
    tfp.layers.DenseFlipout(512, activation=tf.nn.relu),
    tfp.layers.DenseFlipout(10),
])

logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)
