import numpy as np
import tensorflow as tf


class SeasonalEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args, input_dim, period, N, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.period = period
        self.N = N
        # model params
        self.a_n, self.b_n = [tf.keras.layers.Embedding(input_dim=input_dim, output_dim=N, input_length=1)] * 2


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

    @staticmethod
    def fourier_series(a_n, b_n, t, n, period):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )
    

class PooledSeasonalEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args, period, N, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.N = N
        # model params
        self.a_n, self.b_n = [tf.keras.layers.Dense(self.N, activation='linear')] * 2

    def call(self, input_tensor, training=False):
        X, t = input_tensor
        n = (tf.range(self.N, dtype='float') + 1)[None, :]

        return self.fourier_series(
            self.a_n(X),
            self.b_n(X),
            n,
            self.period,
            t)

    @staticmethod
    def fourier_series(a_n, b_n, n, period, t):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )

    
class LinearTrendEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args, t_range, n_changepoints=25, checkpoint_range=.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_range = t_range
        self.n_changepoints = n_changepoints
        self.checkpoint_range = checkpoint_range
        self.r = tf.keras.layers.Reshape((3, self.n_changepoints))
        self.δ = tf.keras.layers.Dense(self.n_changepoints * 3, kernel_regularizer='l1')
        self.m = tf.keras.layers.Dense(1)
        self.k = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.s = tf.cast(tf.linspace(self.t_range[0], int(self.checkpoint_range * self.t_range[1]), self.n_changepoints + 1), 'float')[1:]

    def call(self, input_tensor, training=False):
        X, t = input_tensor
        return self.trend(
            self.r(self.δ(X)),
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
    