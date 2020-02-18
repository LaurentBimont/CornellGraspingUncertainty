import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Lambda, Wrapper, InputSpec


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1]).value   # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        print(type(self.dropout_regularizer), type(input_dim), input_dim)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


def heteroscedastic_loss(true, pred):
    mean = pred[:, :5]
    log_var = pred[:, 5:]
    precision = K.exp(-log_var)
    return K.sum(precision * (true - mean) ** 2. + log_var, -1)


def min_mse(y_true, y_pred):
    y_pred_temp = K.repeat(y_pred, K.shape(y_true)[1])
    G = K.square(y_pred_temp - y_true)
    G = K.mean(G, axis=-1)
    G = K.min(G, axis=-1)
    return G


def neg_log_likelihood(y_true, y_pred):
    my_mean = y_pred[:, :5]
    my_var = y_pred[:, 5:]
    my_mean_temp = K.repeat(my_mean, K.shape(y_true)[1])
    my_var = (K.log(1 + K.exp(my_var))+1e-6)
    numerateur = K.min(K.square(my_mean_temp-y_true), axis=1)
    denominateur = 2*my_var
    result = numerateur*K.pow(denominateur, -1)
    return K.log(K.square(my_var))/2 + result


def numerateur(y_true, y_pred):
    my_mean = y_pred[:, :5]

    my_var = y_pred[:, 5:]
    my_mean_temp = K.repeat(my_mean, K.shape(y_true)[1])
    my_var = (K.log(1 + K.exp(my_var)) + 1e-6)
    print(K.print_tensor(my_var))
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) +\
    #        0.5*K.log(2*np.pi)
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) + 0.5*K.log(2*np.pi)   K.min(K.square(my_mean_temp-y_true), axis=1)/
    numerateur = K.min(K.square(my_mean_temp - y_true), axis=1)
    denominateur = 2*my_var
    return numerateur


def denominateur(y_true, y_pred):
    my_mean = y_pred[:, :5]

    my_var = y_pred[:, 5:]
    my_mean_temp = K.repeat(my_mean, K.shape(y_true)[1])
    my_var = (K.log(1 + K.exp(my_var)) + 1e-6)
    print(K.print_tensor(my_var))
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) +\
    #        0.5*K.log(2*np.pi)
    # return K.mean(K.log(K.square(my_var))/2 + K.min(K.square(my_mean_temp-y_true), axis=1)/(2*K.square(my_var))) + 0.5*K.log(2*np.pi)   K.min(K.square(my_mean_temp-y_true), axis=1)/
    denominateur = 2*my_var
    return denominateur


def value_min(y_true, y_pred):
    return K.min(y_pred)


def value_max(y_true, y_pred):
    return K.max(y_pred)
