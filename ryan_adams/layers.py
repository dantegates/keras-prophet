import numpy as np
import tensorflow as tf
from tensorflow import keras


def _chain_layers(*layers):
    L = layers[0]
    for layer in layers[1:]:
        L = layer(L)
    return L


def _singular_embedding_index(input_batch):
    tf_constant = keras.backend.constant(np.array([[0]]))
    batch_size = keras.backend.shape(input_batch)[0]
    tiled_constant = keras.backend.tile(tf_constant, (batch_size, 1))
    return tiled_constant


class BaseComponentLayer(keras.layers.Layer):
    def __init__(self, *args, n_items,
                 shared_embedding_projection=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_items = n_items
        self.shared_embedding_projection = shared_embedding_projection
        # `keras` will automatically track trainable variables that are stored as
        # instance attributes - even if those attributes are a list.
        # Just use name mangling so we don't have to worry about coming up with
        # a uber-clever name or colliding with a `keras` attribute.
        self.__layers = []

    def _init_parameter(self, dim, name, regularizer=None):
        name = f'{self.name}_{name}'

        flattened_dim = dim if isinstance(dim, int) else np.prod(dim)
        output_dim = (dim,) if isinstance(dim, int) else dim

        layers = [
            keras.layers.Embedding(
                input_dim=self.n_items,
                output_dim=flattened_dim,
                input_length=1,
                embeddings_regularizer=regularizer,
                name=f'{name}_embedding')
        ]

        if self.shared_embedding_projection:
            layers.append(keras.layers.Dense(flattened_dim, name=f'{name}_dense'))

        layers.append(keras.layers.Reshape(output_dim, name=f'{name}_reshape'))

        def param(id, layers=layers):
            return _chain_layers(id, *layers)

        self.__layers.append(layers)
        return param

    def call(self, inputs):
        t, id = inputs
        if id is None:
            if self.n_items > 1:
                raise ValueError
            id = keras.layers.Lambda(_singular_embedding_index)(t)
        return t, id


class LinearTrend(BaseComponentLayer):
    def __init__(self,
                 *args,
                 t0: int = None,
                 t_range: tuple = None,
                 n_changepoints: int = 20,
                 changepoint_range: float = .8,
                 changepoint_penalty: float = 10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = t0
        self.t_range = t_range
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoint_penalty = changepoint_penalty

        self.m = self._init_parameter(1, 'm')
        self.k = self._init_parameter(1, 'k')
        self.δ = self._init_parameter((1, self.n_changepoints), 'delta', keras.regularizers.L1(self.changepoint_penalty))

    def call(self, input_tensor):
        t, id = super().call(input_tensor)
        m = self.m(id)
        k = self.k(id)
        δ = self.δ(id)

        # self.add_loss(keras.regularizers.l1(l1=1e-3)(δ))
        changepoints = tf.linspace(
            self.t_range[0],
            int(self.changepoint_range * self.t_range[1]),
            self.n_changepoints + 1)
        s = tf.cast(changepoints, 'float')[1:]

        return self.trend(m, k, δ, t, s)

    @staticmethod
    def trend(m, k, δ, t, s):
        a_t = tf.convert_to_tensor(t > s)
        A = keras.backend.switch(a_t, tf.ones_like(t), tf.zeros_like(t))
        # cast to float and reshape for compatibility with δ
        A = tf.cast(A, 'float')[:, None, :]

        trend = (tf.reduce_sum(tf.multiply(A, δ), axis=-1) + k) * t

        γ = tf.multiply(-s[None, None, :], δ)
        offset = tf.reduce_sum(tf.multiply(A, γ), axis=-1) + m

        return trend + offset


class SaturatingTrend(LinearTrend):
    def __init__(self, *args, cap, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cap

    def call(self, inputs):
        linear_trend = super().call(inputs)
        return self.cap / (1 + tf.exp(-linear_trend))


class Seasonality(BaseComponentLayer):
    def __init__(self,
                 *args,
                 period: float,
                 order: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.order = order
        self.a_n = self._init_parameter(self.order, 'a_n')
        self.b_n = self._init_parameter(self.order, 'b_n')

    def call(self, input_tensor):
        t, id = super().call(input_tensor)
        a_n = self.a_n(id)
        b_n = self.b_n(id)
        n = (tf.range(self.order, dtype='float') + 1)[None, :]
        return self.fourier_series(a_n, b_n, n, self.period, t)

    @staticmethod
    def fourier_series(a_n, b_n, n, period, t):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )
