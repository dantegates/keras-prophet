import numpy as np
import tensorflow as tf



class SeasonalMixin:
    @staticmethod
    def fourier_series(a_n, b_n, n, period, t):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )



class SeasonalEmbedding(tf.keras.layers.Layer, SeasonalMixin):
    def __init__(self, *args, input_dim, period, N, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.period = period
        self.N = N
        # model params
        self.a_n, self.b_n = [tf.keras.layers.Embedding(input_dim=input_dim, output_dim=N, input_length=1) for _ in range(2)]


    def call(self, input_tensor, training=False):
        id, t = input_tensor
        n = (tf.range(self.N, dtype='float') + 1)[None, :]

        instance_trend = self.fourier_series(
            tf.squeeze(self.a_n(id), axis=1),
            tf.squeeze(self.b_n(id), axis=1),
            t,
            n,
            self.period)
        global_trend = self.fourier_series(
            tf.squeeze(self.a_n(tf.zeros_like(id)), axis=1),
            tf.squeeze(self.b_n(tf.zeros_like(id)), axis=1),
            t,
            n,
            self.period)
        
        return instance_trend + global_trend
    

class CallaborativeSeasonalEmbedding(tf.keras.layers.Layer, SeasonalMixin):
    def __init__(self, *args, period, N, n_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.N = N
        self.n_layers = n_layers
        # model params
        self.a_n = [tf.keras.layers.Dense(self.N, activation='linear')
                    for _ in range(self.n_layers)]
        self.b_n = [tf.keras.layers.Dense(self.N, activation='linear')
                    for _ in range(self.n_layers)]

    def call(self, input_tensor, training=False):
        X, t = input_tensor
        n = (tf.range(self.N, dtype='float') + 1)[None, :]

        a_n = b_n = X
        for W_a, W_b in zip(self.a_n, self.b_n):
            a_n, b_n = W_a(a_n), W_b(b_n)

        return self.fourier_series(
            a_n,
            b_n,
            n,
            self.period,
            t)


class CallaborativeLinearTrendEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args, output_dim, t_range, n_changepoints=25, checkpoint_range=.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.t_range = t_range
        self.n_changepoints = n_changepoints
        self.checkpoint_range = checkpoint_range
        self.r = tf.keras.layers.Reshape((self.output_dim, self.n_changepoints))
        self.δ = tf.keras.layers.Dense(self.n_changepoints * self.output_dim)
        self.m = tf.keras.layers.Dense(self.output_dim)
        self.k = tf.keras.layers.Dense(self.output_dim)

    def build(self, input_shape):
        self.s = tf.cast(tf.linspace(self.t_range[0], int(self.checkpoint_range * self.t_range[1]), self.n_changepoints + 1), 'float')[1:]

    def call(self, input_tensor, training=False):
        X, t = input_tensor

        weights = self.r(self.δ(X))
        self.add_loss(tf.keras.regularizers.l1()(weights))
        return self.trend(
            weights,
            self.m(X),
            self.k(X),
            self.s,
            t)

    @staticmethod
    def trend(δ, m, k, s, t):
        A = tf.cast((t > s), 'float')[:, None, :]

        trend = (tf.reduce_sum(tf.multiply(A, δ), axis=-1) + k) * t

        γ = tf.multiply(-s[None, None, :], δ)
        offset = tf.reduce_sum(tf.multiply(A, γ), axis=-1) + m

        return trend + offset
    